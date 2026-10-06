"""Compare lexical encodings separately from real translation-pipeline counts."""
import argparse
import hashlib
import json
import math
import statistics
import re
from pathlib import Path

from .common import digest, normalize, read_jsonl, write_json, write_jsonl
from .model import TranslationModel


def summary(counts, words):
    values = sorted(counts)
    return dict(total_tokens=sum(counts), total_whitespace_words=sum(words),
                tokens_per_word=sum(counts) / sum(words), median=statistics.median(values),
                p95=values[max(0, math.ceil(0.95 * len(values)) - 1)],
                over_256=sum(x > 256 for x in counts), over_512=sum(x > 512 for x in counts), n=len(counts))


def analyze(rows, cfg, out):
    import importlib.metadata
    import tiktoken
    system = TranslationModel(cfg, tokenizer_only=True)
    encodings = {n: tiktoken.get_encoding(n) for n in ('cl100k_base', 'o200k_base')}
    counts = []
    for row in rows:
        for language, key in [('en', 'source'), ('hi', 'target')]:
            text = normalize(row[key])
            words = len(text.split())
            for name, enc in encodings.items():
                ids = enc.encode(text, disallowed_special=())
                counts.append(dict(pair_id=row['pair_id'], judgment_id=row['judgment_id'], language=language,
                                   tokenizer=name, tokens=len(ids), special_tokens=0, words=words, unknown=0))
            ids, special = system.lexical(text, target=language == 'hi')
            counts.append(dict(pair_id=row['pair_id'], judgment_id=row['judgment_id'], language=language,
                               tokenizer='native', tokens=len(ids), special_tokens=special, words=words,
                               unknown=ids.count(system.tokenizer.unk_token_id)))
    aggregates = {}
    for name in (*encodings, 'native'):
        aggregates[name] = {}
        for lang in ('en', 'hi'):
            subset = [r for r in counts if r['tokenizer'] == name and r['language'] == lang]
            aggregates[name][lang] = dict(summary([r['tokens'] for r in subset], [r['words'] for r in subset]),
                special_tokens=sum(r['special_tokens'] for r in subset), unknown=sum(r['unknown'] for r in subset),
                unknown_fraction=sum(r['unknown'] for r in subset) / sum(r['tokens'] for r in subset))
        aggregates[name]['hindi_english_ratio'] = aggregates[name]['hi']['total_tokens'] / aggregates[name]['en']['total_tokens']
        aggregates[name]['hindi_reduction_vs_cl100k'] = 1 - aggregates[name]['hi']['total_tokens'] / aggregates['cl100k_base']['hi']['total_tokens']
    pipeline = []
    for row in rows:
        proc = system.processor(False)
        source = system.encode(system.preprocess([row['source']], proc))['input_ids'][0]
        target = system.encode(system.preprocess([row['target']], proc, target=True), target=True)['input_ids'][0]
        pipeline.append(dict(pair_id=row['pair_id'], training_source_tokens=len(source), training_target_tokens=len(target),
                             training_source_unknown=source.count(system.tokenizer.unk_token_id),
                             training_target_unknown=target.count(system.tokenizer.unk_token_id),
                             inference_source_tokens=system.inference_lengths([row])[0]))
    terms = {}
    for term in ['न्यायालय', 'अधिकारिता', 'अपील खारिज', 'प्राकृतिक न्याय', 'धारा 9-क(2)', '24.01.2005']:
        ids, _ = system.lexical(term, target=True)
        native = ([system.tokenizer.tgt_decoder.get(i, system.tokenizer.unk_token) for i in ids]
                  if system.indic else system.tokenizer.convert_ids_to_tokens(ids))
        terms[term] = dict(native=native, **{name: [enc.decode_single_token_bytes(i).hex() for i in enc.encode(term)] for name, enc in encodings.items()})
    out = Path(out)
    write_jsonl(out / 'counts.jsonl', counts)
    write_jsonl(out / 'pipeline-counts.jsonl', pipeline)
    write_json(out / 'tokenizers.json', dict(lexical=aggregates, terms=terms, config=cfg,
        content_sha256=hashlib.sha256(json.dumps(rows, ensure_ascii=False, sort_keys=True).encode()).hexdigest(),
        versions={n: importlib.metadata.version(n) for n in ['tiktoken', 'transformers', 'sentencepiece', 'IndicTransToolkit']},
        term_format='Native token pieces; tiktoken pieces as UTF-8 bytes in hex (some tokens split codepoints).',
        note='Identical NFC text, native source/target modes. Pipeline preprocessing is counted separately. Fixed encodings do not change with LoRA.'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', default='data/primary/frozen.jsonl')
    p.add_argument('--config', default='configs/primary.json')
    p.add_argument('--out', default='runs/tokenizers')
    group = p.add_mutually_exclusive_group()
    group.add_argument('--documents', action='store_true', help='Measure all original documents, including visibly corrupted files')
    group.add_argument('--usable-documents', action='store_true', help='Measure documents containing Devanagari, before translation length filtering')
    p.add_argument('--root', default='data/primary')
    args = p.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    excluded = []
    if args.documents or args.usable_documents:
        from .prepare import documents
        docs = documents(args.root)
        rows = [dict(pair_id=str(j), judgment_id=j, source=docs[j, 'en'], target=docs[j, 'hi']) for j in range(1, 31)]
        if args.usable_documents:
            excluded = [dict(judgment_id=r['judgment_id'], reason='No Devanagari in supplied Hindi extraction')
                        for r in rows if not re.search(r'[\u0900-\u097f]', r['target'])]
            rows = [r for r in rows if r['judgment_id'] not in {e['judgment_id'] for e in excluded}]
    else:
        rows = read_jsonl(args.data)
    analyze(rows, cfg, args.out)
    write_json(Path(args.out) / 'sample.json', dict(unit='documents' if args.documents or args.usable_documents else 'aligned passages',
               examples=len(rows), excluded=excluded, selection='Has Devanagari; no model length filter' if args.usable_documents else 'Supplied input',
               data_sha256=digest(args.data) if not (args.documents or args.usable_documents) else None))


if __name__ == '__main__':
    main()
