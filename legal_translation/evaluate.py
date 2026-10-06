"""Generate ordered predictions, reproduce saved metrics, and export blind review."""
import argparse
import json
import random
from pathlib import Path

from .common import digest, normalize, read_jsonl, unique, write_json, write_jsonl


def score(references, predictions):
    from indicnlp.tokenize import indic_tokenize
    from sacrebleu.metrics import BLEU, CHRF
    refs, preds = unique(references), unique(predictions)
    if not refs or refs.keys() != preds.keys():
        raise ValueError('Missing, extra, or mismatched prediction IDs')
    for key, pred in preds.items():
        for field in ('judgment_id', 'source', 'target', 'split'):
            if field in pred and pred[field] != refs[key].get(field):
                raise ValueError(f'Mismatched {field} for {key}')
        if not isinstance(pred['prediction'], str):
            raise ValueError('Predictions must be text')

    def metrics(ids):
        hyp = [' '.join(normalize(preds[k]['prediction']).split()) for k in ids]
        ref = [' '.join(normalize(refs[k]['target']).split()) for k in ids]
        tokenize = lambda texts: [' '.join(indic_tokenize.trivial_tokenize(t, lang='hi')) for t in texts]
        bleu, chrf = BLEU(tokenize='none'), CHRF(char_order=6, word_order=0, beta=2)
        b, c = bleu.corpus_score(tokenize(hyp), [tokenize(ref)]), chrf.corpus_score(hyp, [ref])
        return {'n': len(ids), 'indic_bleu': b.score, 'chrf': c.score,
                'bleu_signature': str(bleu.get_signature()), 'chrf_signature': str(chrf.get_signature()),
                'generation_limit_hits': sum(preds[k].get('hit_generation_limit', False) for k in ids)}

    groups = {str(j): metrics([k for k, r in refs.items() if r['judgment_id'] == j])
              for j in sorted({r['judgment_id'] for r in references})}
    lengths = {}
    for label, low, high in [('1-64', 1, 64), ('65-128', 65, 128), ('129-256', 129, 256)]:
        ids = [k for k in refs if low <= preds[k].get('source_tokens', refs[k].get('source_tokens', 0)) <= high]
        if ids:
            lengths[label] = metrics(ids)
    import importlib.metadata
    result = {'overall': metrics(list(refs)), 'per_judgment': groups, 'source_token_groups': lengths,
            'versions': {n: importlib.metadata.version(n) for n in ('sacrebleu', 'indic-nlp-library-itt')},
            'normalization': 'NFC, whitespace collapse; BLEU Indic trivial_tokenize(hi); chrF detokenized, whitespace ignored'}
    quality_groups = {}
    for label, statuses in [('no_material_flag', {'PASS', 'PASS-context', 'PASS-preparation-repaired'}),
                            ('material_reference_flag', {'FLAG-medium', 'FLAG-high'}), ('minor_reference_flag', {'FLAG-low'})]:
        ids = [k for k, r in refs.items() if r.get('reference_quality', {}).get('status') in statuses]
        if ids:
            quality_groups[label] = metrics(ids)
    if quality_groups:
        result['reference_quality_groups'] = quality_groups
        result['reference_quality_note'] = 'Codex reference inspection, not human validation. Diagnostic groups; all examples retained in overall scores.'
    return result


def review(selection, references, baseline, adapted, out):
    refs, a, b = unique(references), unique(baseline), unique(adapted)
    unique(selection)
    expected = {'terminology': 4, 'complex': 4, 'facts': 4}
    if len(selection) != 12 or {k: sum(s['category'] == k for s in selection) for k in expected} != expected:
        raise ValueError('Review selection requires four examples in each of terminology/complex/facts')
    if not set(s['pair_id'] for s in selection) <= refs.keys() & a.keys() & b.keys():
        raise ValueError('Review selection IDs are missing')
    rng, blind, key = random.Random(42), [], []
    for s in selection:
        pid = s['pair_id']
        systems = ['baseline', 'adapted']; rng.shuffle(systems)
        outputs = {'baseline': a[pid]['prediction'], 'adapted': b[pid]['prediction']}
        blind.append(dict(s, source=refs[pid]['source'], reference=refs[pid]['target'],
                          A=outputs[systems[0]], B=outputs[systems[1]],
                          meaning='', omissions='', additions='', terminology='', factual_details='', fluency='', reference_defects=''))
        key.append(dict(pair_id=pid, A=systems[0], B=systems[1]))
    write_jsonl(Path(out) / 'review-blind.jsonl', blind)
    write_json(Path(out) / 'review-key.json', key)
    sections = ['# Hindi review\n\nRate against the English source. The supplied Hindi reference may contain errors. '
                'Keep review-key.json closed until all 12 examples are rated.\n\n'
                'For A and B separately, rate meaning, terminology, factual details and fluency from 0 to 4: '
                '0 unusable; 1 major errors; 2 partly correct; 3 minor errors; 4 correct. '
                'Record omitted or added facts and mark errors minor, major or critical. '
                'A reversed outcome, party identity or negation is critical. Do not reward fluency that changes meaning.\n']
    for index, row in enumerate(blind, 1):
        sections.append(f'## {index}. {row["pair_id"]} ({row["category"]})\n\n'
                        f'English:\n\n{row["source"]}\n\nSupplied reference:\n\n{row["reference"]}\n\n'
                        f'A:\n\n{row["A"]}\n\nB:\n\n{row["B"]}\n\n'
                        'A ratings: meaning __/4; terminology __/4; facts __/4; fluency __/4.\n\n'
                        'B ratings: meaning __/4; terminology __/4; facts __/4; fluency __/4.\n\n'
                        'Omissions/additions and severity:\n\nReference defects:\n\nPreferred output: A / B / tie. Reason:\n')
    (Path(out) / 'review.md').write_text('\n'.join(sections), encoding='utf-8')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    a = sub.add_parser('score'); a.add_argument('--data', default='data/primary/frozen.jsonl'); a.add_argument('--split', choices=['dev', 'test'], required=True); a.add_argument('--predictions', required=True); a.add_argument('--out', required=True)
    a = sub.add_parser('predict'); a.add_argument('--data', default='data/primary/frozen.jsonl'); a.add_argument('--split', choices=['dev', 'test'], required=True); a.add_argument('--adapter'); a.add_argument('--out', required=True); a.add_argument('--config', default='configs/primary.json')
    a = sub.add_parser('infer'); a.add_argument('--input', required=True); a.add_argument('--output', required=True); a.add_argument('--adapter'); a.add_argument('--config', default='configs/primary.json')
    a = sub.add_parser('review'); a.add_argument('--data', default='data/primary/frozen.jsonl'); a.add_argument('--selection', default='data/primary/review-selection.jsonl'); a.add_argument('--baseline', required=True); a.add_argument('--adapted', required=True); a.add_argument('--out', default='runs/review')
    args = p.parse_args()
    if args.command == 'score':
        refs = [r for r in read_jsonl(args.data) if r['split'] == args.split]
        result = score(refs, read_jsonl(args.predictions))
        result.update(data_sha256=digest(args.data), predictions_sha256=digest(args.predictions))
        write_json(args.out, result)
    elif args.command == 'review':
        review(read_jsonl(args.selection), read_jsonl(args.data), read_jsonl(args.baseline), read_jsonl(args.adapted), args.out)
    else:
        from .model import TranslationModel
        cfg = json.loads(Path(args.config).read_text())
        if args.command == 'predict':
            from .model import verify_frozen
            verify_frozen(args.data, cfg)
        model = TranslationModel(cfg, adapter=args.adapter,
                                 data_sha256=digest(args.data) if args.command == 'predict' else None)
        if args.command == 'infer':
            import re
            text = normalize(Path(args.input).read_text(encoding='utf-8-sig'))
            rows = [dict(pair_id=str(i), source=s) for i, s in enumerate(re.split(r'\n\s*\n', text)) if s.strip()]
            if not rows:
                raise ValueError('Empty input')
            predictions, stats = model.predict(rows)
            if any(r['hit_generation_limit'] for r in predictions):
                raise ValueError('Generation limit reached; output may be incomplete. Split the input at checked paragraph boundaries.')
            unverified = {r['pair_id']: r['unverified_source_dates'] for r in predictions if r['unverified_source_dates']}
            if unverified:
                raise ValueError(f'Date transfer could not be verified; output withheld for human review: {unverified}')
            Path(args.output).write_text('\n\n'.join(r['prediction'] for r in predictions) + '\n', encoding='utf-8')
            write_json(args.output + '.metadata.json', stats)
        else:
            rows = [r for r in read_jsonl(args.data) if r['split'] == args.split]
            predictions, stats = model.predict(rows)
            write_jsonl(args.out, predictions)
            write_json(args.out + '.metadata.json', dict(stats, data_sha256=digest(args.data), config=cfg,
                adapter=str(args.adapter) if args.adapter else None,
                adapter_sha256=model.adapter_metadata['adapter_sha256'] if args.adapter else None))
            write_json(args.out + '.metrics.json', score(rows, predictions))


if __name__ == '__main__':
    main()
