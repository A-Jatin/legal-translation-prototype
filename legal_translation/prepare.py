"""Acquire immutable text, propose boundaries, apply reviewed bilingual spans."""
import argparse
import hashlib
import json
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

from .common import digest, normalize, read_jsonl, unique, write_json, write_jsonl


def split_for(judgment):
    if judgment not in range(1, 31):
        raise ValueError(f'Unknown judgment: {judgment}')
    return 'dev' if judgment in (8, 9, 24) else 'test' if judgment in (1, 4, 21) else 'train'


def acquire(sources, root):
    import requests
    root = Path(root)
    entries = json.loads(Path(sources).read_text())
    expected = {(i, lang) for i in range(1, 31) for lang in ('en', 'hi')}
    if {(e['judgment_id'], e['language']) for e in entries} != expected or len(entries) != 60:
        raise ValueError('Source manifest must contain exactly 30 bilingual judgments')
    old = { (e['judgment_id'], e['language']): e for e in read_jsonl(root / 'manifest.jsonl')} if (root / 'manifest.jsonl').exists() else {}

    def fetch(e):
        path = root / 'raw' / e['language'] / f"{e['judgment_id']}.txt"
        previous = old.get((e['judgment_id'], e['language']))
        if path.exists():
            if not previous or digest(path) != previous['sha256']:
                raise ValueError(f'Unmanifested or modified immutable input: {path}')
            raw = path.read_bytes()
            entry = previous
        else:
            response = requests.get(e['url'], timeout=60)
            response.raise_for_status()
            raw = response.content
            if not raw or b'<html' in raw[:300].lower() or b'<!doctype html' in raw[:300].lower():
                raise ValueError(f'Expected text, got an HTML/download error: {e["source_id"]}')
            raw.decode('utf-8-sig')
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
            entry = dict(e, path=str(path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(),
                         downloaded_at=datetime.now(timezone.utc).isoformat(), source_last_modified=response.headers.get('Last-Modified'))
        derived = root / 'derived' / e['language'] / path.name
        derived.parent.mkdir(parents=True, exist_ok=True)
        derived.write_text(normalize(raw.decode('utf-8-sig')), encoding='utf-8')
        return entry

    # Preserve a manifest after every successful download so interrupted runs resume safely.
    records = []
    for source in entries:
        entry = fetch(source)
        records.append(entry)
        write_jsonl(root / 'manifest.jsonl', records + [v for k, v in old.items() if k not in {(r['judgment_id'], r['language']) for r in records}])
    write_json(root / 'splits.json', {str(i): split_for(i) for i in range(1, 31)})
    return records


def documents(root):
    return {(i, lang): (Path(root) / 'derived' / lang / f'{i}.txt').read_text(encoding='utf-8')
            for i in range(1, 31) for lang in ('en', 'hi')}


def boundaries(text):
    # Suggestions only: nested numbering and quotations require bilingual review.
    starts = sorted({0, *[m.start() for m in re.finditer(r'(?m)^(?:[0-9०-९]+[.)]|\([ivxlcdm]+\))\s', text)]})
    return [[a, b] for a, b in zip(starts, starts[1:] + [len(text)]) if text[a:b].strip()]


def align(docs, corrections):
    unique(corrections)
    rows, exclusions, used = [], [], {}
    for c in corrections:
        j = c['judgment_id']
        split = split_for(j)
        if c['alignment_status'] != 'verified':
            if not c.get('note'):
                raise ValueError('Unverified/excluded spans require a reason')
            exclusions.append(dict(c, split=split, reason=c['note']))
            continue
        texts = {}
        for lang, key in [('en', 'source'), ('hi', 'target')]:
            text = docs[j, lang]
            start, end = c[key + '_span']
            if not (0 <= start < end <= len(text)):
                raise ValueError(f'Invalid span in {c["pair_id"]}')
            word_char = lambda ch: unicodedata.category(ch)[0] in 'LMN'
            for boundary in (start, end):
                if 0 < boundary < len(text) and word_char(text[boundary - 1]) and word_char(text[boundary]):
                    raise ValueError(f'Alignment boundary inside a word in {c["pair_id"]} {lang}')
            spans = used.setdefault((j, lang), [])
            if any(start < b and a < end for a, b in spans):
                raise ValueError(f'Overlapping retained spans in {c["pair_id"]}')
            spans.append((start, end))
            texts[key] = normalize(text[start:end])
            if not texts[key]:
                raise ValueError('Empty aligned text')
        rows.append(dict(c, split=split, **texts))
    return rows, exclusions


def deduplicate(rows):
    canonical = lambda s: re.sub(r'^\d+[.)]\s*', '', ' '.join(s.split()))
    heldout = {canonical(r['source']) for r in rows if r['split'] != 'train'}
    kept, removed = [], []
    for r in rows:
        if r['split'] == 'train' and canonical(r['source']) in heldout:
            removed.append(dict(r, reason='exact_heldout_source'))
        else:
            kept.append(r)
    return kept, removed


def overlap_audit(rows):
    def shingles(text):
        words = re.findall(r'\w+', normalize(text).casefold())
        return {tuple(words[i:i + 5]) for i in range(len(words) - 4)}

    train = [(r, shingles(r['source'])) for r in rows if r['split'] == 'train']
    heldout = [(r, shingles(r['source'])) for r in rows if r['split'] != 'train']
    overlaps = []
    for h, hs in heldout:
        for t, ts in train:
            common = hs & ts
            if len(common) >= 2:
                overlaps.append(dict(heldout_id=h['pair_id'], train_id=t['pair_id'],
                                     shared_five_word_shingles=len(common),
                                     jaccard=len(common) / len(hs | ts)))
    return sorted(overlaps, key=lambda x: (-x['shared_five_word_shingles'], x['heldout_id'], x['train_id']))


def coverage(docs, rows):
    return {str(j): {lang: {'total_characters': len(docs[j, lang]),
                           'retained_characters': sum(r[key + '_span'][1] - r[key + '_span'][0] for r in rows if r['judgment_id'] == j),
                           'pairs': sum(r['judgment_id'] == j for r in rows)}
                    for lang, key in [('en', 'source'), ('hi', 'target')]}
            for j in range(1, 31)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    a = sub.add_parser('acquire'); a.add_argument('--sources', default='data/primary/sources.json'); a.add_argument('--root', default='data/primary')
    a = sub.add_parser('propose'); a.add_argument('--root', default='data/primary')
    a = sub.add_parser('align'); a.add_argument('--root', default='data/primary'); a.add_argument('--corrections', default='data/primary/corrections.jsonl')
    a = sub.add_parser('freeze'); a.add_argument('--root', default='data/primary'); a.add_argument('--config', default='configs/primary.json'); a.add_argument('--out')
    args = p.parse_args()
    if args.command == 'acquire':
        print(f'Acquired {len(acquire(args.sources, args.root))} files')
        return
    docs = documents(args.root)
    if args.command == 'freeze':
        from .model import TranslationModel
        cfg = json.loads(Path(args.config).read_text())
        model = TranslationModel(cfg, tokenizer_only=True)
        path = Path(args.out) if args.out else Path(args.root) / 'frozen.jsonl'
        if path.exists():
            raise ValueError('Evaluation is already frozen; do not overwrite it')
        rows = read_jsonl(Path(args.root) / 'aligned.jsonl')
        excluded = read_jsonl(Path(args.root) / 'exclusions.jsonl')
        retained = []
        for row in rows:
            try:
                feature = model.features([row])[0]
                inference_length = model.inference_lengths([row])[0]
                if inference_length > cfg['token_limit']:
                    raise ValueError('Inference source exceeds working token limit')
            except ValueError as e:
                excluded.append(dict(row, reason='oversized', detail=str(e)))
                continue
            original = {}
            for lang, key in [('en', 'source'), ('hi', 'target')]:
                raw = (Path(args.root) / 'raw' / lang / f'{row["judgment_id"]}.txt').read_bytes().decode('utf-8')
                pattern = r'\s+'.join(re.escape(t) for t in row[key].split())
                matches = list(re.finditer(pattern, raw))
                if len(matches) != 1:
                    raise ValueError(f'Original span is ambiguous for {row["pair_id"]} {lang}; record a manual raw span')
                m = matches[0]
                original[key] = {'path': f'raw/{lang}/{row["judgment_id"]}.txt', 'character_span': [m.start(), m.end()],
                                 'text': raw[m.start():m.end()]}
            retained.append(dict(row, original_spans=original, source_tokens=len(feature['input_ids']),
                                 target_tokens=len(feature['labels']), inference_source_tokens=inference_length))
        for split, expected in [('dev', {8, 9, 24}), ('test', {1, 4, 21})]:
            if {r['judgment_id'] for r in retained if r['split'] == split} != expected:
                raise ValueError(f'Working token limit excluded a whole {split} judgment; inspect and split before freezing')
        write_jsonl(path, retained)
        write_jsonl(path.with_suffix('.exclusions.jsonl') if args.out else Path(args.root) / 'exclusions.jsonl', excluded)
        write_json(path.with_suffix('.coverage.json') if args.out else Path(args.root) / 'coverage.json', coverage(docs, retained))
        write_json(str(path) + '.metadata.json', dict(sha256=digest(path), model_id=cfg['model_id'], revision=cfg['revision'],
            token_limit=cfg['token_limit'], frozen_at=datetime.now(timezone.utc).isoformat(),
            corrections_sha256=digest(Path(args.root) / 'corrections.jsonl'), manifest_sha256=digest(Path(args.root) / 'manifest.jsonl'),
            architecture_limit=model.architecture_limit, normalization='UTF-8 BOM removal, LF line endings, NFC, within-line whitespace collapse'))
        print(f'Frozen {len(retained)} pairs; {len(excluded)} exclusions')
        return
    if args.command == 'propose':
        write_jsonl(Path(args.root) / 'boundaries.jsonl', [dict(judgment_id=j, language=lang, spans=boundaries(text)) for (j, lang), text in docs.items()])
        return
    rows, excluded = align(docs, read_jsonl(args.corrections))
    rows, duplicates = deduplicate(rows)
    write_jsonl(Path(args.root) / 'aligned.jsonl', rows)
    write_jsonl(Path(args.root) / 'exclusions.jsonl', excluded + duplicates)
    write_json(Path(args.root) / 'coverage.json', coverage(docs, rows))
    write_json(Path(args.root) / 'overlap.json', overlap_audit(rows))
    print(f'{len(rows)} reviewed pairs; {len(excluded) + len(duplicates)} excluded')


if __name__ == '__main__':
    main()
