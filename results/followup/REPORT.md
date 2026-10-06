# Follow-up: more development gain, little test improvement

The primary adapter barely changed the observed legal errors. I ran three small follow-ups to test whether different batching, more training passages or more LoRA projections would help. Development scores increased, but the selected adapter did not improve test performance consistently.

| Run | Training pairs | LoRA projections | Best dev chrF | Selected step |
|---|---:|---|---:|---:|
| A | 63 | q/k | 58.21 | 32 |
| B | 145 | q/k | 58.64 | 32 |
| C | 145 | q/k/v/out | 59.06 | 48 |

Baseline development chrF is 57.49; the primary adapter reaches 57.89. All runs use rank 8, alpha 16, dropout 0.05, AdamW 2e-4, microbatch 2 and accumulation 2. They stop at 48 optimizer steps or three epochs, evaluate development at steps 16/32/48 and keep the earlier checkpoint on ties. A completes three epochs; B/C stop at epoch 1.301. Their exposure differs at the same cap, so B versus C does not isolate projection count alone.

The [expanded subset](../../data/expanded/frozen.jsonl) adds 82 checked passages from the supplied corpus, giving 145 training pairs from 16 judgments. All original 120 rows, including the 57 development/test rows, are unchanged. Training source coverage rises from 9.15% to 25.81%; held-out coverage remains fixed. Maximum processed source/target lengths are 250/252, within the 256-token limit. [Corrections](../../data/expanded/corrections.jsonl) and [exclusions](../../data/expanded/exclusions.jsonl) preserve the boundary decisions. Alignment observations are Codex inspection, not human certification; known minor reference defects remain.

## What happened on test

C was selected by development chrF before new test generation. A/B were not ranked on test. The test set was already known from the preceding work.

| System | Test chrF | Test BLEU |
|---|---:|---:|
| Pretrained baseline | 61.24 | 37.41 |
| Primary adapter | 61.68 | 37.87 |
| Follow-up C | 61.75 | 37.42 |

C gains only 0.07 test chrF over the primary adapter and loses 0.45 BLEU. In the 16 unflagged or preparation-repaired references, baseline/C chrF is 66.54/66.25 and BLEU is 46.16/44.81. These are diagnostic annotations; all 31 examples remain in the main score.

C corrects the dismissed case from `याचिका` to `अपील` in `j21-p11`. It still mistranslates “Leave granted” and “Assailing,” loses the salary interval and entitled party, and omits Nem Singh and the finding of illegality. It also newly loses both occurrences of 29.10.2015 in `j01-p06`. Literal-date flags increase from two passages/five occurrences to three passages/six occurrences. [Nine complete examples](qualitative-examples.jsonl) show those gains and failures. These findings give little reason to replace the primary adapter.

Five outputs are unchanged from baseline; 26 change. No output reaches the generation limit. BLEU and chrF use the same normalization and Hindi tokenization as the primary report. [Baseline](baseline-test.jsonl.metrics.json) and [C](adapted-test.jsonl.metrics.json) metrics include judgment/length groups and signatures.

Synchronized test inference takes 76.93 seconds for baseline and 87.17 for C after warmup, excluding loading. Generated tokens are 3,220/3,223 and source padding is 2.65% for both. These single observations do not establish a stable speed difference. Vocabulary and fixed reference token counts are unchanged.

## Reproduce the follow-up

Run from the project root after the setup in [README.md](../../README.md). The selected C adapter is included. A/B development predictions and selection records are retained; their unselected adapter weights are omitted.

```sh
python -m legal_translation.evaluate score --data data/expanded/frozen.jsonl --split test --predictions results/followup/adapted-test.jsonl --out runs/C-scores.json
python -m legal_translation.tokenization --config configs/followup-B.json --data data/expanded/frozen.jsonl --out runs/expanded-tokenizers
python -m legal_translation.train smoke --config configs/followup-C.json --data data/expanded/frozen.jsonl --out runs/C-smoke
python -m legal_translation.train adapt --config configs/followup-C.json --data data/expanded/frozen.jsonl --smoke runs/C-smoke/smoke.json --out runs/C
python -m legal_translation.evaluate predict --config configs/followup-C.json --data data/expanded/frozen.jsonl --split test --adapter runs/C/adapter --out runs/C-test.jsonl
```

Use fresh output paths. A uses `configs/followup-A.json` with `data/primary/frozen.jsonl`; B uses `configs/followup-B.json` with the expanded data. The [A](A/selection.json), [B](B/selection.json) and [C](C/selection.json) records save configuration, checkpoint choice and hashes. Expanded data hashes to `255e74e9ccc22d1a4a1fc54c7844727f5160b6179e7b164f87a5048cab8bef51`; C weights hash to `9d2ff9fce86510a33229212ebc914d408ab668362303bde882807a2723394dea`.

More training and projections raised development matching without repairing the important legal failures. Before another run, I would review complete bilingual clauses and select on source-based factual/terminology checks as well as chrF. The existing blind review remains pending. Any later generalization claim needs an unseen benchmark; this known test set should stay fixed.
