"""Real-model smoke gate and one bounded LoRA adaptation run."""
import argparse
import copy
import json
import math
import time
from pathlib import Path

from .common import digest, read_jsonl, write_json, write_jsonl
from .model import TranslationModel, synchronize, verify_frozen


def seed_everything(seed):
    from transformers import set_seed
    set_seed(seed)


def add_adapter(model, cfg):
    from peft import LoraConfig, TaskType, get_peft_model
    return get_peft_model(model, LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM,
                          r=cfg['rank'], lora_alpha=cfg['alpha'], lora_dropout=cfg['dropout'],
                          target_modules=cfg.get('target_modules', ['q_proj', 'k_proj'])))


def seq2seq_collator(tokenizer, net):
    from transformers import DataCollatorForSeq2Seq
    standard = DataCollatorForSeq2Seq(tokenizer, model=net, label_pad_token_id=-100)

    def collate(batch):
        # Indic source encoding and generation reset this mutable property to left.
        # Set it on every call, including the first training batch after dev scoring.
        tokenizer.padding_side = 'right'
        return standard(batch)

    return collate


def smoke(cfg, rows, output, data_path):
    import torch
    from peft import PeftModel
    seed_everything(cfg['seed'])
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    system = TranslationModel(cfg)
    train = [r for r in rows if r['split'] == 'train']
    dev = [r for r in rows if r['split'] == 'dev']
    if len(train) < 2 or not dev:
        raise ValueError('Smoke gate requires training and development examples')
    features = system.features(train)
    indices = sorted(range(len(train)), key=lambda i: len(features[i]['labels']))
    chosen = [indices[0], indices[-1]]
    if len(features[chosen[0]]['labels']) == len(features[chosen[1]]['labels']):
        raise ValueError('Smoke test requires variable-length labels')
    probes = [train[i] for i in chosen]
    initial, initial_time = system.predict(probes)
    write_jsonl(output / 'baseline-probes.jsonl', initial)
    restoration = 'not applicable: Marian has no entity replacement'
    if system.indic:
        proc = system.processor(True)
        dates = ['The order dated 12.03.2020 awards Rs. 1,250.50 under Section 9-A(2).',
                 'The appeal dated 24.01.2005 cites (2004) 7 SCC 525 and Rs. 9,999.00.']
        processed = system.preprocess(dates, proc)
        restored = proc.postprocess_batch([s.split(' ', 2)[2] for s in processed], lang='eng_Latn')
        for expected, observed in zip(dates, restored):
            for entity in ('12.03.2020', '1,250.50') if '12.03.2020' in expected else ('24.01.2005', '9,999.00'):
                if entity not in observed.replace(', ', ','):
                    raise AssertionError('Ordered date/amount restoration failed')
        restoration = {'inputs': dates, 'restored': restored}
    parameter_count = sum(p.numel() for p in system.model.parameters())
    system.model = add_adapter(system.model, cfg)
    collator = seq2seq_collator(system.tokenizer, system.model)
    batch = collator([copy.deepcopy(features[i]) for i in chosen])
    assert (batch['labels'] == -100).any(), 'Variable-length labels must be masked'
    for labels in batch['labels']:
        valid = labels[labels != -100]
        assert torch.equal(labels[:len(valid)], valid), 'Target padding must be on the right'
    if 'decoder_input_ids' in batch:
        assert (batch['decoder_input_ids'] >= 0).all(), 'Masked label IDs leaked to decoder'
    else:
        # IndicTrans2 shifts labels inside forward and replaces -100 with pad IDs.
        assert system.indic and system.model.config.decoder_start_token_id is not None
    batch = {k: v.to(system.device) for k, v in batch.items()}
    trainable = {n: p for n, p in system.model.named_parameters() if p.requires_grad}
    assert trainable and all('lora_' in n for n in trainable)
    optimizer = torch.optim.AdamW(trainable.values(), lr=cfg['learning_rate'])
    losses = []
    system.model.train()
    synchronize(system.device)
    start = time.perf_counter()
    for step in range(20):
        optimizer.zero_grad(set_to_none=True)
        loss = system.model(**batch).loss
        assert torch.isfinite(loss).item(), 'Nonfinite smoke loss'
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all().item() for p in trainable.values()), 'Missing/nonfinite adapter gradients'
        assert any(torch.count_nonzero(p.grad).item() for p in trainable.values()), 'All adapter gradients zero'
        optimizer.step()
        losses.append(loss.item())
        print(f'smoke optimizer step {step + 1}/20 loss={loss.item():.4f}', flush=True)
    synchronize(system.device)
    seconds = time.perf_counter() - start
    system.model.eval()
    before, _ = system.predict(probes)
    adapter = output / 'discarded-smoke-adapter'
    system.model.save_pretrained(adapter)
    write_json(adapter / 'experiment.json', cfg)
    # Unload smoke adapter, then load its saved parameters on the same unchanged base.
    base = system.model.unload()
    system.model = PeftModel.from_pretrained(base, adapter).to(system.device)
    after, _ = system.predict(probes)
    assert [r['prediction'] for r in before] == [r['prediction'] for r in after], 'Adapter save/reload output mismatch'
    dev_predictions, dev_time = system.predict(dev)
    write_jsonl(output / 'adapted-probes.jsonl', after)
    write_jsonl(output / 'smoke-dev.jsonl', dev_predictions)
    steps = min(cfg['step_limit'], 3 * math.ceil(len(train) / (cfg['microbatch'] * cfg['accumulation'])))
    # Smoke optimizer steps use one microbatch; recorded optimizer steps accumulate gradients.
    evaluations = math.ceil(steps / cfg['evaluation_steps']) if cfg.get('evaluation_steps') else 3
    projected = seconds / 20 * cfg['accumulation'] * steps + dev_time['seconds'] * (1 + evaluations)
    result = dict(status='passed', config=cfg, data_sha256=digest(data_path), base_parameters=parameter_count,
                  adapter_parameters=sum(p.numel() for p in trainable.values()), losses=losses,
                  optimizer_steps=20, seconds=seconds, device=system.device,
                  ordered_restoration=restoration, adapter_reload_equal=True,
                  projected_training_and_dev_seconds=projected, recommended_hardware='accelerator' if projected > 10800 else 'local',
                  baseline_probe_timing=initial_time, dev_timing=dev_time,
                  note='Smoke updates are discarded. Recorded training reloads base model and resets seed.')
    write_json(output / 'smoke.json', result)
    return result


def adapt(cfg, rows, data_path, out, smoke_path):
    import torch
    from datasets import Dataset
    from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, TrainerCallback, get_linear_schedule_with_warmup
    from .evaluate import score
    evidence = json.loads(Path(smoke_path).read_text())
    if evidence['status'] != 'passed' or evidence['config'] != cfg or evidence['data_sha256'] != digest(data_path):
        raise ValueError('Smoke gate missing or config changed; rerun smoke')
    if evidence['recommended_hardware'] == 'accelerator' and not torch.cuda.is_available():
        raise ValueError('Measured runtime requires a faster accelerator; rerun smoke on the intended device')
    out = Path(out)
    if (out / 'selection.json').exists():
        raise ValueError('Recorded experiment already exists; choose a new output directory')
    seed_everything(cfg['seed'])
    system = TranslationModel(cfg)
    if evidence['device'] != system.device:
        raise ValueError('Hardware differs from validated smoke gate')
    train = [r for r in rows if r['split'] == 'train']
    dev = [r for r in rows if r['split'] == 'dev']
    baseline, timing = system.predict(dev)
    write_jsonl(out / 'baseline-dev.jsonl', baseline)
    baseline_scores = score(dev, baseline)
    write_json(out / 'baseline-dev.metrics.json', dict(baseline_scores, timing=timing))
    seed_everything(cfg['seed'])
    system.model = add_adapter(system.model, cfg)
    features = system.features(train)
    log, padding = [], {'slots': 0, 'actual': 0, 'target_slots': 0, 'target_actual': 0}
    standard = seq2seq_collator(system.tokenizer, system.model)

    def collate(batch):
        result = standard(batch)
        padding['slots'] += result['input_ids'].numel()
        padding['actual'] += int(result['attention_mask'].sum().item())
        padding['target_slots'] += result['labels'].numel()
        padding['target_actual'] += int((result['labels'] != -100).sum().item())
        return result

    class Checkpoint(TrainerCallback):
        best = -math.inf
        last_step = -1
        best_path = None

        def on_step_end(self, args, state, control, **kwargs):
            if cfg.get('evaluation_steps') and state.global_step % cfg['evaluation_steps'] == 0:
                self.save_and_score(state)
            if state.global_step >= cfg['step_limit']:
                control.should_training_stop = True
            return control

        def save_and_score(self, state):
            if state.global_step == self.last_step:
                return
            self.last_step = state.global_step
            path = out / f'checkpoint-{state.global_step}'
            predictions, timing = system.predict(dev)
            scores = score(dev, predictions)
            system.model.save_pretrained(path)
            write_json(path / 'experiment.json', dict(cfg, data_sha256=digest(data_path),
                adapter_sha256=digest(path / 'adapter_model.safetensors'),
                adapter_config_sha256=digest(path / 'adapter_config.json')))
            write_jsonl(path / 'dev.jsonl', predictions)
            write_json(path / 'metrics.json', dict(scores, timing=timing))
            row = dict(step=state.global_step, epoch=state.epoch, chrf=scores['overall']['chrf'], checkpoint=str(path))
            log.append(row)
            write_json(out / 'checkpoints.json', log)
            if row['chrf'] > self.best:  # strict comparison preserves earlier checkpoint on ties
                self.best, self.best_path = row['chrf'], path
                system.model.save_pretrained(out / 'adapter')
                write_json(out / 'adapter' / 'experiment.json', dict(cfg, data_sha256=digest(data_path),
                    adapter_sha256=digest(out / 'adapter' / 'adapter_model.safetensors'),
                    adapter_config_sha256=digest(out / 'adapter' / 'adapter_config.json')))
            print(json.dumps(row), flush=True)

        def on_epoch_end(self, args, state, control, **kwargs):
            if not cfg.get('evaluation_steps'):
                self.save_and_score(state)

        def on_train_end(self, args, state, control, **kwargs):
            self.save_and_score(state)  # includes final partial epoch if the step cap stops training

    checkpoint = Checkpoint()
    args = Seq2SeqTrainingArguments(output_dir=str(out), num_train_epochs=3, max_steps=-1,
        per_device_train_batch_size=cfg['microbatch'], gradient_accumulation_steps=cfg['accumulation'],
        learning_rate=cfg['learning_rate'], lr_scheduler_type='linear', warmup_ratio=0.05,
        optim='adamw_torch', weight_decay=0, save_strategy='no', eval_strategy='no',
        logging_strategy='steps', logging_steps=1, group_by_length=True,
        fp16=False, bf16=False, dataloader_pin_memory=False, report_to='none',
        seed=cfg['seed'], data_seed=cfg['seed'], use_cpu=system.device == 'cpu')
    scheduled_steps = min(cfg['step_limit'], 3 * math.ceil(math.ceil(len(features) / cfg['microbatch']) / cfg['accumulation']))
    optimizer = torch.optim.AdamW((p for p in system.model.parameters() if p.requires_grad), lr=cfg['learning_rate'], weight_decay=0)
    scheduler = get_linear_schedule_with_warmup(optimizer, math.ceil(0.05 * scheduled_steps), scheduled_steps)
    trainer = Seq2SeqTrainer(model=system.model, args=args, train_dataset=Dataset.from_list(features),
                             processing_class=system.tokenizer, data_collator=collate, callbacks=[checkpoint],
                             optimizers=(optimizer, scheduler))
    start = time.perf_counter()
    trainer.train()
    synchronize(system.device)
    if trainer.state.global_step > cfg['step_limit'] or trainer.state.epoch > 3:
        raise AssertionError('Training exceeded the experiment budget')
    write_json(out / 'trainer-log.json', trainer.state.log_history)
    write_json(out / 'selection.json', dict(config=cfg, data_sha256=digest(data_path),
        adapter_sha256=digest(out / 'adapter' / 'adapter_model.safetensors'),
        adapter_config_sha256=digest(out / 'adapter' / 'adapter_config.json'),
        best_checkpoint=str(checkpoint.best_path), development_chrf=checkpoint.best,
        baseline_development_chrf=baseline_scores['overall']['chrf'],
        preferred_system='adapted' if checkpoint.best > baseline_scores['overall']['chrf'] else 'baseline',
        selection='highest dev chrF; strict ties select earlier checkpoint; baseline preferred on no gain',
        optimizer_steps=trainer.state.global_step, epochs=trainer.state.epoch,
        training_and_dev_seconds=time.perf_counter() - start, padding=padding))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['smoke', 'adapt'])
    p.add_argument('--config', default='configs/primary.json')
    p.add_argument('--data', default='data/primary/frozen.jsonl')
    p.add_argument('--out')
    p.add_argument('--smoke', default='runs/smoke/smoke.json')
    args = p.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    rows = verify_frozen(args.data, cfg)
    if args.command == 'smoke':
        smoke(cfg, rows, args.out or 'runs/smoke', args.data)
    else:
        adapt(cfg, rows, args.data, args.out or 'runs/adaptation', args.smoke)


if __name__ == '__main__':
    main()
