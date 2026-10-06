# Legal translation: English to Hindi

A small IndicTrans2 experiment on supplied court judgments. The primary LoRA adapter scores **61.68 chrF / 37.87 BLEU**, against a pretrained baseline of **61.24 / 37.41**. Read [the report](REPORT.md) for the tokenizer comparison, legal errors and limits of those gains.

## Repository

- `legal_translation/`: preparation, tokenizer analysis, training and evaluation/inference.
- `configs/`: the primary experiment and three bounded follow-up runs.
- `data/primary/`: original text, normalization, reviewed alignments and the frozen primary dataset.
- `data/expanded/`: the larger training subset used by the follow-up; evaluation membership is unchanged.
- `results/primary/`: predictions, metrics, tokenizer counts, development checkpoints and the selected adapter.
- `results/followup/`: development comparisons, test results and the selected C adapter.
- `tests/`: data contracts, scoring, padding, adapter integrity and bounded-training checks.

## Setup

Run commands from the repository root. The validated local setup uses Python 3.11.15, the exact packages in `requirements.lock`, and Mac MPS with FP32/eager attention.

```sh
uv venv --python 3.11.15 .venv
uv pip sync --python .venv/bin/python requirements.lock
source .venv/bin/activate
```

Saved scoring needs no model download or GPU. For tokenization, model tests, training or inference, accept [IndicTrans2 model access](https://huggingface.co/ai4bharat/indictrans2-en-indic-dist-200M), then authenticate:

```sh
hf auth login
export PYTHONHASHSEED=42
export HF_MODULES_CACHE="$PWD/.cache/modules"
export TIKTOKEN_CACHE_DIR="$PWD/.cache/tiktoken"
python -m unittest discover -s tests -v
```

Tests download pinned model code/configuration and tokenizer assets on first use, not pretrained weights. After a successful online run, `HF_HUB_OFFLINE=1` tests offline reuse. Tiktoken uses its own cache; retain it for offline tokenizer analysis. Model weights are downloaded when translation or training first runs. Credentials and caches are excluded from Git.

## Recompute the saved scores

```sh
python -m legal_translation.evaluate score --split test --predictions results/primary/baseline-test.jsonl --out runs/baseline-scores.json
python -m legal_translation.evaluate score --split test --predictions results/primary/adapted-test.jsonl --out runs/adapted-scores.json
```

Compare metric fields with the saved `*.metrics.json` files. The CLI also records input hashes. Missing, duplicate or mismatched prediction IDs fail scoring.

## Prepare and compare tokenizers

Raw and derived inputs are included. Acquisition verifies existing immutable files; it does not download them again.

```sh
python -m legal_translation.prepare acquire
python -m legal_translation.prepare align
python -m legal_translation.prepare freeze --out runs/primary-frozen.jsonl
python -m legal_translation.tokenization --out runs/passage-tokenizers
python -m legal_translation.tokenization --usable-documents --out runs/document-tokenizers
```

The reproduced frozen JSONL must hash to `ecc99aa202b32d8552e47068224041cbf9d2a1fd7b3ce4d81caf6fb9d3cf3cba`. Freezing refuses to overwrite an existing output. The unfiltered document comparison excludes only the five Hindi extractions with no Devanagari. Required model tokens and preprocessing are counted separately from lexical encoding.

## Train and translate

The CLI defaults to `configs/primary.json` and `data/primary/frozen.jsonl`. Use fresh output directories for each run.

```sh
python -m legal_translation.train smoke --out runs/smoke
python -m legal_translation.train adapt --smoke runs/smoke/smoke.json --out runs/adaptation
python -m legal_translation.evaluate predict --split test --out runs/baseline-test.jsonl
python -m legal_translation.evaluate predict --split test --adapter runs/adaptation/adapter --out runs/adapted-test.jsonl
```

Smoke updates are discarded. Training reloads the base model, resets seed 42 and selects on development chrF only. The primary run selected step 12. Its saved adapter is `results/primary/adaptation/adapter`.

For English paragraphs separated by blank lines:

```sh
python -m legal_translation.evaluate infer --input english.txt --output hindi.txt --adapter results/primary/adaptation/adapter
```

Output preserves paragraph order. Oversized input, a generation-limit hit or an unverified literal source date causes an error before writing. The date guard cannot check party identity, date roles, negation, legal outcomes or all factual additions. Passing it does not establish a faithful translation.

The [12 blind review examples](results/primary/review/review.md) are ready for a bilingual reviewer; their ratings remain blank. [The follow-up report](results/followup/REPORT.md) gives the separate A/B/C configurations and reproduction commands. Generated runs stay under ignored `runs/`; reported results remain in `results/`.
