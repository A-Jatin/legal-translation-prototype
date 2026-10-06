"""Small regression checks for data leakage, provenance and scoring contracts."""
import unittest

from legal_translation import prepare
from legal_translation import evaluate
from legal_translation import model
import tempfile
from pathlib import Path
import json
import hashlib
from unittest.mock import Mock, patch


def tiny_model(cfg):
    import torch
    from transformers import AutoConfig
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    kwargs = dict(revision=cfg['revision'], cache_dir=cfg['cache_dir'])
    config = AutoConfig.from_pretrained(cfg['model_id'], trust_remote_code=True, **kwargs)
    for k, v in dict(encoder_layers=1, decoder_layers=1, encoder_embed_dim=16,
                    decoder_embed_dim=16, encoder_ffn_dim=32, decoder_ffn_dim=32,
                    encoder_attention_heads=2, decoder_attention_heads=2,
                    encoder_vocab_size=32, decoder_vocab_size=32, vocab_size=32,
                    dropout=0., attention_dropout=0., activation_dropout=0.).items():
        setattr(config, k, v)
    config._attn_implementation = 'eager'
    torch.manual_seed(42)
    cls = get_class_from_dynamic_module(config.auto_map['AutoModelForSeq2SeqLM'], cfg['model_id'], **kwargs)
    return cls(config).eval()


class DataChecks(unittest.TestCase):
    def test_split_and_normalization(self):
        self.assertEqual([i for i in range(1, 31) if prepare.split_for(i) == 'dev'], [8, 9, 24])
        self.assertEqual([i for i in range(1, 31) if prepare.split_for(i) == 'test'], [1, 4, 21])
        self.assertEqual(prepare.normalize('\ufeffA\r\n  B\t C\n\nD'), 'A\nB C\n\nD')

    def test_explicit_spans_not_line_counts(self):
        docs = {(2, 'en'): '1. Alpha.\n(i) Beta.', (2, 'hi'): '१. अल्फा।\n(i) बीटा।'}
        correction = dict(pair_id='2-a', judgment_id=2, source_span=[0, 9], target_span=[0, 9],
                          alignment_status='verified', note='Manually checked boundary')
        rows, excluded = prepare.align(docs, [correction])
        self.assertEqual(rows[0]['source'], '1. Alpha.')
        self.assertFalse(excluded)
        with self.assertRaises(ValueError):
            prepare.align(docs, [correction, correction])
        with self.assertRaises(ValueError):
            prepare.align(docs, [dict(correction, source_span=[0, 900])])

    def test_exact_heldout_passage_removed(self):
        rows = [dict(pair_id='a', split='train', source='The  appeal is allowed.'),
                dict(pair_id='b', split='test', source='The appeal is allowed.')]
        kept, removed = prepare.deduplicate(rows)
        self.assertEqual([r['pair_id'] for r in kept], ['b'])
        self.assertEqual(removed[0]['reason'], 'exact_heldout_source')
        numbered = [dict(pair_id='train-numbered', split='train', source='2. A few facts need mention hereinbelow.'),
                    dict(pair_id='dev-numbered', split='dev', source='3. A few facts need mention hereinbelow.')]
        retained, removed = prepare.deduplicate(numbered)
        self.assertEqual([r['pair_id'] for r in retained], ['dev-numbered'])
        overlap = prepare.overlap_audit([
            dict(pair_id='t', split='train', source='The appeal was allowed by the court.'),
            dict(pair_id='h', split='test', source='The appeal was allowed by the court today.')])
        self.assertEqual(overlap[0]['shared_five_word_shingles'], 3)

    def test_scoring_rejects_bad_ids_and_preserves_reference(self):
        refs = [dict(pair_id='a', judgment_id=1, source='An appeal.', target='अपील खारिज।')]
        preds = [dict(pair_id='a', prediction='अपील खारिज ।')]
        scores = evaluate.score(refs, preds)
        self.assertAlmostEqual(scores['overall']['chrf'], 100)
        self.assertEqual(scores['overall']['indic_bleu'], 0)  # fewer than four n-grams
        self.assertEqual(refs[0]['target'], 'अपील खारिज।')
        with self.assertRaises(ValueError):
            evaluate.score(refs, preds * 2)
        with self.assertRaises(ValueError):
            evaluate.score(refs, [dict(pair_id='b', prediction='')])

    def test_hindi_punctuation_bleu(self):
        refs = [dict(pair_id='a', judgment_id=1, source='The appeal was dismissed.',
                     target='यह अपील न्यायालय द्वारा खारिज की जाती है।')]
        predictions = [dict(pair_id='a', prediction='यह अपील न्यायालय द्वारा खारिज की जाती है ।')]
        self.assertAlmostEqual(evaluate.score(refs, predictions)['overall']['indic_bleu'], 100)

    def test_reference_diagnostics_do_not_drop_overall_examples(self):
        refs = [dict(pair_id='a', judgment_id=1, source='A.', target='यह अपील खारिज की जाती है।',
                     reference_quality={'status': 'PASS'}),
                dict(pair_id='b', judgment_id=1, source='B.', target='यह याचिका स्वीकार की जाती है।',
                     reference_quality={'status': 'FLAG-high'})]
        preds = [dict(pair_id=r['pair_id'], prediction=r['target']) for r in refs]
        result = evaluate.score(refs, preds)
        self.assertEqual(result['overall']['n'], 2)
        self.assertEqual(result['reference_quality_groups']['no_material_flag']['n'], 1)
        self.assertEqual(result['reference_quality_groups']['material_reference_flag']['n'], 1)
        self.assertEqual(result['overall']['chrf'], 100)

    def test_adapter_requires_matching_dataset_and_weight_hash(self):
        with tempfile.TemporaryDirectory() as temp:
            adapter = Path(temp)
            (adapter / 'adapter_model.safetensors').write_bytes(b'adapter A')
            (adapter / 'adapter_config.json').write_text('{}')
            cfg = dict(model_id=model.FALLBACK, revision='a' * 40, token_limit=256,
                       generation_limit=256, num_beams=4)
            metadata = dict(cfg, data_sha256='dataset A',
                            adapter_sha256=hashlib.sha256(b'adapter A').hexdigest(),
                            adapter_config_sha256=hashlib.sha256(b'{}').hexdigest())
            (adapter / 'experiment.json').write_text(json.dumps(metadata))
            self.assertEqual(model.verify_adapter(adapter, cfg, 'dataset A')['adapter_sha256'], metadata['adapter_sha256'])
            with self.assertRaises(ValueError):
                model.verify_adapter(adapter, cfg, 'dataset B')
            (adapter / 'adapter_model.safetensors').write_bytes(b'adapter B')
            with self.assertRaises(ValueError):
                model.verify_adapter(adapter, cfg, 'dataset A')

    def test_oversized_inference_fails_before_generation(self):
        system = model.TranslationModel.__new__(model.TranslationModel)
        system.cfg = {'token_limit': 256}
        system.inference_lengths = lambda rows: [257]
        with self.assertRaisesRegex(ValueError, 'Oversized source input'):
            system.predict([{'pair_id': 'long', 'source': 'very long text'}])

    def test_date_transfer_requires_each_literal_occurrence(self):
        source = 'Orders dated 12.03.2020 and 12.03.2020 were set aside.'
        self.assertEqual(model.unverified_source_dates(source, 'आदेश 12.03.2020 को अपास्त हुआ।'), ['12.03.2020'])
        self.assertEqual(model.unverified_source_dates(source, '12.03.2020 तथा 12.03.2020'), [])
        self.assertEqual(model.unverified_source_dates('No date here.', 'कोई तारीख नहीं।'), [])

    def test_alignment_rejects_cut_inside_date(self):
        docs = {(21, 'en'): 'The sale occurred.', (21, 'hi'): 'बिक्री 13.04.2006 को हुई।'}
        correction = dict(pair_id='sale', judgment_id=21, source_span=[0, 18],
                          target_span=[0, docs[21, 'hi'].index('4.2006')], alignment_status='verified')
        with self.assertRaisesRegex(ValueError, 'inside a word'):
            prepare.align(docs, [correction])

    def test_adapter_rejects_modified_configuration(self):
        from legal_translation.common import digest
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            (path / 'adapter_model.safetensors').write_bytes(b'weights')
            (path / 'adapter_config.json').write_text(json.dumps({'lora_alpha': 16}))
            cfg = dict(model_id=model.FALLBACK, revision='a' * 40, token_limit=256,
                       generation_limit=256, num_beams=4)
            metadata = dict(cfg, adapter_sha256=digest(path / 'adapter_model.safetensors'),
                            adapter_config_sha256=digest(path / 'adapter_config.json'))
            (path / 'experiment.json').write_text(json.dumps(metadata))
            model.verify_adapter(path, cfg)
            (path / 'adapter_config.json').write_text(json.dumps({'lora_alpha': 16000}))
            with self.assertRaisesRegex(ValueError, 'configuration hash'):
                model.verify_adapter(path, cfg)

    def test_native_source_tags_are_overhead(self):
        system = model.TranslationModel(json.loads(Path('configs/primary.json').read_text()), tokenizer_only=True)
        text = 'The appeal stands dismissed.'
        source_ids, source_special = system.lexical(text)
        tagged = system.encode(['eng_Latn hin_Deva ' + text], special=False)['input_ids'][0]
        self.assertEqual(source_ids, tagged[2:])
        self.assertEqual(source_special, 3)
        _, target_special = system.lexical('अपील खारिज की जाती है।', target=True)
        self.assertEqual(target_special, 1)

    def test_target_padding_keeps_valid_decoder_logits(self):
        import copy
        import sys
        import torch
        from legal_translation import train
        cfg = json.loads(Path('configs/primary.json').read_text())
        system = model.TranslationModel(cfg, tokenizer_only=True)
        tiny = tiny_model(cfg)
        collate = train.seq2seq_collator(system.tokenizer, tiny)
        short = dict(input_ids=[5, 6, 2], attention_mask=[1, 1, 1], labels=[10, 11, 2])
        long = dict(short, labels=[10, 11, 12, 13, 14, 2])
        # Inference changes tokenizer mode between epochs; every collation must correct it.
        for _ in range(2):
            system.encode(['eng_Latn hin_Deva source'])
            batch = collate(copy.deepcopy([short, long]))
            self.assertEqual(batch['labels'][0].tolist(), [10, 11, 2, -100, -100, -100])
            shift = sys.modules[type(tiny).__module__].shift_tokens_right
            self.assertEqual(shift(batch['labels'], 1, 2)[0, :3].tolist(), [2, 10, 11])
            with torch.no_grad():
                plain = tiny(input_ids=torch.tensor([[5, 6, 2]]), attention_mask=torch.ones(1, 3),
                             labels=torch.tensor([[10, 11, 2]]), use_cache=False).logits
                padded = tiny(**batch, use_cache=False).logits
            torch.testing.assert_close(plain[0], padded[0, :3], rtol=1e-5, atol=1e-6)

    def test_download_resume_html_and_immutable_input(self):
        sources = [dict(judgment_id=j, language=lang, source_id=f'{j}-{lang}', url='https://example.invalid/text')
                   for j in range(1, 31) for lang in ['en', 'hi']]
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source_path = root / 'sources.json'
            source_path.write_text(json.dumps(sources))
            response = Mock(content='\ufeffपहला\r\n पाठ'.encode('utf-8'), headers={})
            with patch('requests.get', side_effect=[response, RuntimeError('download interrupted')]):
                with self.assertRaisesRegex(RuntimeError, 'interrupted'):
                    prepare.acquire(source_path, root)
            self.assertEqual(len(prepare.read_jsonl(root / 'manifest.jsonl')), 1)
            with patch('requests.get', return_value=response) as fetch:
                self.assertEqual(len(prepare.acquire(source_path, root)), 60)
                self.assertEqual(fetch.call_count, 59)
            self.assertEqual((root / 'derived/en/1.txt').read_text(), 'पहला\nपाठ')
            (root / 'raw/en/1.txt').write_text('modified')
            with patch('requests.get') as fetch:
                with self.assertRaisesRegex(ValueError, 'modified immutable'):
                    prepare.acquire(source_path, root)
                fetch.assert_not_called()
            html = Mock(content=b'<!DOCTYPE html><html>error</html>', headers={})
            with patch('requests.get', return_value=html):
                with self.assertRaisesRegex(ValueError, 'Expected text'):
                    prepare.acquire(source_path, root / 'html')
            self.assertFalse((root / 'html/raw/en/1.txt').exists())

    def test_partial_epoch_stop_and_checkpoint_ties(self):
        from legal_translation import train
        from legal_translation.common import digest, write_json, write_jsonl
        cfg = dict(json.loads(Path('configs/primary.json').read_text()), device='cpu', step_limit=5)
        tokenizer = model.TranslationModel(cfg, tokenizer_only=True).tokenizer

        class TinySystem:
            device = 'cpu'
            def __init__(self, cfg):
                self.model, self.tokenizer = tiny_model(cfg), tokenizer
            def predict(self, rows):
                return [dict(r, prediction=r['target']) for r in rows], {}
            def features(self, rows):
                return [dict(input_ids=[5, 6, 2], attention_mask=[1, 1, 1], labels=[10, 11, 2]) for r in rows]

        rows = [dict(pair_id=str(i), judgment_id=2, split='train', source='The appeal.',
                     target='यह अपील खारिज की जाती है।') for i in range(64)]
        rows.append(dict(rows[0], pair_id='dev', judgment_id=8, split='dev'))
        for cap, interval, steps in [(5, None, [4, 5]), (10, 3, [3, 6, 9, 10])]:
            config = dict(cfg, step_limit=cap)
            if interval:
                config['evaluation_steps'] = interval
            with self.subTest(interval=interval), tempfile.TemporaryDirectory() as folder:
                root = Path(folder); data = root / 'data.jsonl'; smoke = root / 'smoke.json'; out = root / 'adaptation'
                write_jsonl(data, rows)
                write_json(smoke, dict(status='passed', config=config, data_sha256=digest(data),
                                      recommended_hardware='local', device='cpu'))
                with patch('legal_translation.train.TranslationModel', TinySystem):
                    train.adapt(config, rows, data, out, smoke)
                selection = json.loads((out / 'selection.json').read_text())
                self.assertEqual(selection['optimizer_steps'], cap)
                self.assertLess(selection['epochs'], 3)
                self.assertEqual([r['step'] for r in json.loads((out / 'checkpoints.json').read_text())], steps)
                self.assertTrue(selection['best_checkpoint'].endswith(f'checkpoint-{steps[0]}'))
                self.assertEqual(selection['preferred_system'], 'baseline')
                model.verify_adapter(out / 'adapter', config, digest(data))

    def test_explicit_lora_targets(self):
        from legal_translation import train
        cfg = json.loads(Path('configs/primary.json').read_text())
        for modules in [['q_proj', 'k_proj'], ['q_proj', 'k_proj', 'v_proj', 'out_proj']]:
            with self.subTest(modules=modules):
                net = train.add_adapter(tiny_model(cfg), dict(cfg, target_modules=modules))
                self.assertEqual(net.peft_config['default'].target_modules, set(modules))
                self.assertTrue(all(not p.requires_grad or 'lora_' in n for n, p in net.named_parameters()))

    def test_pinned_generation_cache_requires_workaround(self):
        import torch
        cfg = json.loads(Path('configs/primary.json').read_text())
        tiny = tiny_model(cfg)
        batch = dict(input_ids=torch.tensor([[5, 6, 2]]), attention_mask=torch.ones(1, 3))
        with self.assertRaises(AttributeError):
            tiny.generate(**batch, num_beams=4, max_new_tokens=4, use_cache=True)
        result = tiny.generate(**batch, num_beams=4, max_new_tokens=4, use_cache=False)
        self.assertEqual(result.shape[0], 1)
        self.assertGreater(result.shape[1], 1)


if __name__ == '__main__':
    unittest.main()
