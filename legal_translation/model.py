"""Pinned model loading and state-safe IndicTrans2/Marian preprocessing."""
import copy
import json
import platform
import re
import time
from collections import Counter
from pathlib import Path

from .common import digest, read_jsonl, unique

PRIMARY = 'ai4bharat/indictrans2-en-indic-dist-200M'
PRIMARY_REVISION = '173b94239f7c38886b2747b8d4a5db771a7e1232'
FALLBACK = 'Helsinki-NLP/opus-mt-en-hi'
DATE = re.compile(r'(?<!\d)\d{1,2}[./-]\d{1,2}[./-]\d{4}(?!\d)')


def unverified_source_dates(source, translation):
    # Exact transfer is required for unattended inference; verbalized dates need review.
    return list((Counter(DATE.findall(source)) - Counter(DATE.findall(translation))).elements())


def synchronize(device):
    import torch
    if device == 'mps':
        torch.mps.synchronize()
    elif device.startswith('cuda'):
        torch.cuda.synchronize()


def verify_frozen(path, cfg):
    metadata = json.loads(Path(str(path) + '.metadata.json').read_text())
    if metadata['sha256'] != digest(path):
        raise ValueError('Frozen dataset was modified')
    for key in ('model_id', 'revision', 'token_limit'):
        if metadata[key] != cfg[key]:
            raise ValueError(f'Frozen data/model mismatch: {key}')
    rows = read_jsonl(path)
    unique(rows)
    for split, expected in [('dev', {8, 9, 24}), ('test', {1, 4, 21})]:
        if {r['judgment_id'] for r in rows if r['split'] == split} != expected:
            raise ValueError(f'Missing fixed {split} judgment')
    from .prepare import split_for
    if any(r['split'] != split_for(r['judgment_id']) or r['alignment_status'] != 'verified' for r in rows):
        raise ValueError('Invalid split or unverified alignment')
    return rows


def verify_adapter(path, cfg, data_sha256=None):
    path = Path(path)
    metadata = json.loads((path / 'experiment.json').read_text())
    if any(metadata.get(k) != cfg[k] for k in ('model_id', 'revision', 'token_limit', 'generation_limit', 'num_beams')):
        raise ValueError('Adapter experiment/model configuration mismatch')
    if data_sha256 is not None and metadata.get('data_sha256') != data_sha256:
        raise ValueError('Adapter was trained on a different frozen dataset')
    if metadata.get('adapter_sha256') != digest(path / 'adapter_model.safetensors'):
        raise ValueError('Adapter weight hash mismatch')
    if not (path / 'adapter_config.json').exists() or metadata.get('adapter_config_sha256') != digest(path / 'adapter_config.json'):
        raise ValueError('Adapter configuration hash missing or mismatched; use a fully recorded adapter')
    return metadata


class TranslationModel:
    def __init__(self, cfg, adapter=None, tokenizer_only=False, data_sha256=None):
        import torch
        from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer
        self.cfg = cfg
        self.adapter_metadata = None
        self.indic = cfg['model_id'] == PRIMARY
        if cfg['model_id'] not in (PRIMARY, FALLBACK) or len(cfg['revision']) != 40:
            raise ValueError('Use a supported model pinned to an immutable 40-character commit')
        if self.indic and cfg['revision'] != PRIMARY_REVISION:
            raise ValueError('IndicTrans2 revision differs from the plan')
        kwargs = dict(revision=cfg['revision'], trust_remote_code=self.indic, cache_dir=cfg.get('cache_dir', '.cache/huggingface'))
        self.tokenizer = AutoTokenizer.from_pretrained(cfg['model_id'], use_fast=False, **kwargs)
        config = AutoConfig.from_pretrained(cfg['model_id'], **kwargs)
        limit = getattr(config, 'max_position_embeddings', 1024)
        for key in ('max_source_positions', 'max_target_positions'):
            limit = min(limit, getattr(config, key, limit))
        if cfg['token_limit'] > limit or cfg['generation_limit'] + 1 > limit:
            raise ValueError(f'Working/generation limit exceeds architecture limit {limit}')
        self.architecture_limit = limit
        self.device = cfg.get('device', 'auto')
        if self.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        self.model = None
        if not tokenizer_only:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(cfg['model_id'], dtype=torch.float32, attn_implementation='eager', **kwargs).to(self.device)
            if adapter:
                self.adapter_metadata = verify_adapter(adapter, cfg, data_sha256)
                from peft import PeftModel
                self.model = PeftModel.from_pretrained(self.model, adapter).to(self.device)
            self.model.eval()

    def processor(self, inference):
        if not self.indic:
            return None
        from IndicTransToolkit.processor import IndicProcessor
        return IndicProcessor(inference=inference)

    def preprocess(self, texts, processor, target=False):
        if processor is None:
            return texts
        return processor.preprocess_batch(texts, src_lang='hin_Deva' if target else 'eng_Latn',
                                          tgt_lang='eng_Latn' if target else 'hin_Deva', is_target=target)

    def encode(self, texts, target=False, special=True):
        kwargs = dict(add_special_tokens=special, truncation=False, padding=False)
        return self.tokenizer(text_target=texts, **kwargs) if target else self.tokenizer(texts, **kwargs)

    def features(self, rows):
        processor = self.processor(False)
        src = self.encode(self.preprocess([r['source'] for r in rows], processor))
        tgt = self.encode(self.preprocess([r['target'] for r in rows], processor, target=True), target=True)
        result = []
        for i, row in enumerate(rows):
            if max(len(src['input_ids'][i]), len(tgt['input_ids'][i])) > self.cfg['token_limit']:
                raise ValueError(f'Oversized pair {row["pair_id"]}; split at checked bilingual boundaries')
            result.append(dict(input_ids=src['input_ids'][i], attention_mask=src['attention_mask'][i], labels=tgt['input_ids'][i]))
        return result

    def inference_lengths(self, rows):
        # A throwaway processor cannot contaminate the restoration queue used by generation.
        proc = self.processor(True)
        return [len(ids) for ids in self.encode(self.preprocess([r['source'] for r in rows], proc))['input_ids']]

    def lexical(self, text, target=False):
        # Source language tags are emitted even when add_special_tokens=False.
        tagged = f'eng_Latn hin_Deva {text}' if self.indic and not target else text
        ids = self.encode([tagged], target=target, special=False)['input_ids'][0]
        full = self.encode([tagged], target=target, special=True)['input_ids'][0]
        if self.indic and not target:
            ids = ids[2:]
        return ids, len(full) - len(ids)

    def predict(self, rows, warmup=True):
        import torch
        if not rows:
            raise ValueError('No examples to generate')
        unique(rows)
        lengths = self.inference_lengths(rows)
        if max(lengths) > self.cfg['token_limit']:
            raise ValueError('Oversized source input; checked splitting is required, truncation is disabled')
        ordered = sorted(zip(rows, lengths), key=lambda pair: (pair[1], pair[0]['pair_id']))
        processor = self.processor(True)
        was_training = self.model.training
        self.model.eval()
        batch_size = self.cfg['inference_batch_size']
        all_predictions, generated_count, slots, actual = [], 0, 0, 0
        generation = copy.deepcopy(self.model.generation_config)
        generation.num_beams = self.cfg['num_beams']
        generation.do_sample = False
        generation.max_new_tokens = self.cfg['generation_limit']
        # ponytail: legacy IndicTrans2 cache format breaks with Transformers 4.57; revisit if throughput requires cached decoding.
        generation.use_cache = not self.indic

        def batch_predict(batch, proc):
            sources = self.preprocess([r['source'] for r, _ in batch], proc)
            encoded = self.tokenizer(sources, truncation=False, padding=True, return_tensors='pt').to(self.device)
            with torch.inference_mode():
                outputs = self.model.generate(**encoded, generation_config=generation)
            text = self.tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            if proc is not None:
                text = proc.postprocess_batch(text, lang='hin_Deva')
            return outputs, text, encoded

        if warmup:
            batch_predict(ordered[:min(batch_size, len(ordered))], self.processor(True))
        synchronize(self.device)
        start = time.perf_counter()
        try:
            for offset in range(0, len(ordered), batch_size):
                batch = ordered[offset:offset + batch_size]
                outputs, translations, encoded = batch_predict(batch, processor)
                slots += encoded['input_ids'].numel()
                actual += int(encoded['attention_mask'].sum().item())
                for (row, length), output, translation in zip(batch, outputs.tolist(), translations):
                    # Encoder-decoder output begins with a decoder-start token; count through first EOS.
                    sequence = output[1:]
                    eos = self.tokenizer.eos_token_id
                    count = sequence.index(eos) + 1 if eos in sequence else len(sequence)
                    hit = count >= self.cfg['generation_limit']
                    generated_count += count
                    all_predictions.append(dict(row, prediction=translation, source_tokens=length,
                                                generated_tokens=count, hit_generation_limit=hit,
                                                unverified_source_dates=unverified_source_dates(row['source'], translation)))
            synchronize(self.device)
            seconds = time.perf_counter() - start
        finally:
            self.model.train(was_training)
        by_id = unique(all_predictions)
        return [by_id[r['pair_id']] for r in rows], {
            'seconds': seconds, 'generated_tokens': generated_count, 'tokens_per_second': generated_count / seconds,
            'examples_per_second': len(rows) / seconds, 'source_padding_fraction': (slots - actual) / slots,
            'device': self.device, 'platform': platform.platform(), 'torch': torch.__version__,
            'inference_batch_size': batch_size, 'precision': 'float32', 'attention': 'eager',
            'timing': 'after warmup; synchronized preprocess+generate+postprocess; excludes loading and length sorting',
            'num_beams': self.cfg['num_beams'], 'generation_limit': self.cfg['generation_limit'],
            'examples_with_unverified_dates': sum(bool(r['unverified_source_dates']) for r in all_predictions),
        }
