---
base_model: ai4bharat/indictrans2-en-indic-dist-200M
library_name: peft
tags:
- base_model:adapter:ai4bharat/indictrans2-en-indic-dist-200M
- lora
- transformers
---

# Run C: selected legal translation adapter

English-to-Hindi LoRA adaptation of `ai4bharat/indictrans2-en-indic-dist-200M`, revision `173b94239f7c38886b2747b8d4a5db771a7e1232`. Training used 145 aligned passages from the supplied judgments and `q_proj, k_proj, v_proj, out_proj` projections, with rank 8, alpha 16 and dropout 0.05. The tokenizer and base embeddings are unchanged.

The selected checkpoint is step 48, with development chrF 59.06. [Selection](../selection.json) records the checkpoint and scores. [experiment.json](experiment.json) records the configuration and adapter hashes.

Use `python -m legal_translation.evaluate` with `configs/followup-C.json` and this adapter directory. The [follow-up report](../../REPORT.md) gives reproduction commands and the legal errors observed. Human Hindi ratings remain pending; development gains do not establish legal fidelity.

PEFT 0.21.0.
