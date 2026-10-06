# English-to-Hindi legal translation

Hindi tokenization improved substantially when using an Indic tokenizer. Legal translation improved only slightly after adaptation. These are separate findings: choosing a tokenizer with fewer tokens did not, by itself, demonstrate better translation.

I used IndicTrans2 as the pretrained baseline and adapted it with LoRA on a small set of court-judgment passages. On 31 test passages, the adapter gained 0.44 chrF and 0.46 BLEU. Several errors that could change a reader's understanding of a judgment remained unchanged.

## What changed after adaptation

| System | chrF | BLEU |
|---|---:|---:|
| Pretrained IndicTrans2 | 61.24 | 37.41 |
| Legal-domain LoRA adapter | 61.68 | 37.87 |

Nineteen of the 31 outputs are identical. The remaining changes produced the small aggregate gain, but they did not consistently repair legal meaning. For example:

- **“Leave granted”** becomes `छुट्टी दी गई` in both systems. Permission to appeal is rendered as time off.
- **“Assailing the decision”** becomes `निर्णय को स्वीकार करते हुए` in both. Challenging a decision becomes accepting it.
- In the salary dispute, adaptation restores **29.10.2015**, but both outputs still lose the final payment interval and assign entitlement to the wrong party.
- In the appointment dispute, both outputs lose **Nem Singh**, the allegation that he was prevented from joining, and the court's final finding that the method was unsustainable in law.

There are also passages that retain the relevant conditions. Both systems preserve the four-month appointment deadline, temporary continuation, Article 142 and salary until regular appointment in `j04-p16`. For the archaic phrase “catena of decisions,” both use `निर्णयों के एक समूह`; the sufficient-cause condition survives, while “Limitation Act” becomes `सीमा अधिनियम` and needs terminology review.

[Eight complete examples](results/primary/qualitative-examples.jsonl) contain the English, supplied reference, both outputs and observations. These observations are Codex inspection. The [12 blind examples](results/primary/review/review.md) have not yet received human Hindi ratings. I would not infer improved legal fidelity from the automatic scores alone.

## Where the token savings come from

I compared `cl100k_base`, `o200k_base` and IndicTrans2's native tokenizer on the same normalized text. The main sample contains all 25 usable English/Hindi judgment pairs, before excluding passages for model length. Hindi word counts use whitespace separation.

| Encoding | Hindi tokens | Tokens per Hindi word | Reduction against cl100k |
|---|---:|---:|---:|
| cl100k_base | 195,452 | 5.096 | — |
| o200k_base | 64,775 | 1.689 | 66.86% |
| IndicTrans2 native | 48,985 | 1.277 | 74.94% |

For `न्यायालय`, the three encodings use 9, 4 and 1 tokens respectively. For `अधिकारिता`, they use 11, 4 and 2. This illustrates how the Indic vocabulary reduces fragmentation of Hindi legal terms.

All 25 complete judgments still exceed both 256 and 512 tokens with every encoding. A more economical tokenizer does not remove the need to split long judgments. On the 120 retained passages, 65 exceed 256 Hindi tokens under cl100k, three under o200k, and none under the native lexical encoding.

The native source tokenizer adds two language tags and EOS; the target adds EOS. Those tokens are excluded from the table. Raw native encoding also has unknowns: 4.74% of English tokens and 0.14% of Hindi tokens in this sample. The actual preprocessed pipeline has zero unknowns on the same sample. These separate measurements prevent preprocessing or unknown-token substitution from appearing to be lossless compression. [Full counts](results/primary/document-tokenizers/tokenizers.json) include English/Hindi ratios, passage lengths, unknowns, versions and term pieces; [passage counts](results/primary/tokenizers/tokenizers.json) cover the frozen dataset.

At an equal hypothetical price per token, o200k uses 33.14% of cl100k's Hindi token budget and native uses 25.06%. There was no paid inference or measured billing. I kept IndicTrans2's native tokenizer and embeddings: substituting another vocabulary would require compatible embeddings and training. The comparison establishes token economy, not a controlled tokenizer-only effect on BLEU or chrF.

## Data used, and what was left out

The supplied corpus contains 30 bilingual judgments, or 60 text files. Originals are unchanged; [the manifest](data/primary/manifest.jsonl) records source IDs, retrieval times, sizes and hashes. Preparation removes BOMs, normalizes line endings and Unicode, and cleans whitespace without rewriting names, citations, dates or legal wording.

The split is fixed by judgment: 24 training, three development and three test. Development judgments are 8, 9 and 24; test judgments are 1, 4 and 21. The primary subset contains **63 training, 26 development and 31 test passages**. Only five training judgments contribute retained passages to this small run. Five supplied Hindi files contain no Devanagari and mostly question marks, so they are excluded.

Alignment uses reviewed bilingual paragraphs or checked subspans, never matching line counts. Exact training duplicates of held-out passages are removed. Ambiguous and oversized passages have explicit exclusion reasons. The 256-token acceptance limit includes required preprocessing and special tokens; nothing is silently truncated. [Corrections](data/primary/corrections.jsonl), [exclusions](data/primary/exclusions.jsonl) and [coverage](data/primary/coverage.json) expose those choices.

The retained set covers 16.49% of all English source text, including 47.83% of test text. Test judgment 21 contributes only 12.02% of its text because much of its long reasoning falls outside the accepted spans. Scores therefore describe this subset, not complete-document translation.

The supplied references also contain errors. Thirteen test passages have material reference flags and two have minor flags; all 31 remain in the main score. On the 16 unflagged or preparation-repaired references, chrF changes from 66.54 to 66.79 and BLEU from 46.16 to 46.44. These diagnostic annotations were made after test inspection and do not create an independent clean benchmark. In one reference, statute year 1981 becomes 1984, so a correct model date can be penalized.

One earlier boundary selection cut `j21-p03` inside a date. I repaired the target span from [516, 845] to [516, 913], using the complete supplied paragraph. This is the only held-out target change; the 57 development/test IDs and English sources remain fixed. [The reference policy](data/primary/reference-policy.json) records that correction and seven training-reference exclusions. The test set was already inspected before this repair experiment, so it is a known test set.

## How the run was controlled

The model is [IndicTrans2 en-indic dist-200M](https://huggingface.co/ai4bharat/indictrans2-en-indic-dist-200M), pinned to revision `173b94239f7c38886b2747b8d4a5db771a7e1232`. LoRA trains 884,736 parameters, approximately 0.42% of the base model, on q/k projections with rank 8, alpha 16 and dropout 0.05.

Training uses AdamW at 2e-4, linear decay with 5% warmup, microbatch 2 and accumulation 8. Seed 42, three epochs and an independent 250-step cap are fixed. This subset completes three epochs in 12 optimizer steps. Development chrF is 57.49 for the baseline and 57.44, 57.64 and 57.89 at steps 4, 8 and 12. Step 12 is selected; ties prefer the earlier checkpoint, and baseline remains preferred if adaptation fails to improve development chrF. [Selection and training records](results/primary/adaptation/selection.json) are included.

The run uses FP32/eager attention on Mac MPS. Before training, a real-model smoke test checks finite loss and adapter gradients, label masking, ordered entity restoration and save/reload equality. Its updates are discarded. Right-side target padding is forced for every batch because the native tokenizer changes padding state during generation. Separate processors complete preprocessing, generation and postprocessing for each batch before the next, preserving the entity-restoration queue. Generation caching is disabled for compatibility with the pinned model and Transformers version.

Baseline and adapter use four deterministic beams, batch size 2 and at most 255 new tokens. No test output reaches that limit. BLEU applies Hindi Indic tokenization to both predictions and references before SacreBLEU with `tokenize="none"`; chrF uses normalized detokenized text, character order 6, word order 0 and beta 2. Saved [baseline](results/primary/baseline-test.jsonl.metrics.json) and [adapted](results/primary/adapted-test.jsonl.metrics.json) metrics include signatures, judgment scores and source-length groups. Missing, duplicate or mismatched prediction IDs fail evaluation.

After warmup, synchronized test inference takes 81.08 seconds for baseline and 86.50 for adaptation, excluding model loading. These are single observations. Generated decoder tokens increase from 3,220 to 3,231; source padding is 2.65% for both. LoRA changes neither the vocabulary nor fixed reference token counts, so this run does not improve tokenizer efficiency.

## What the follow-up showed

Three bounded follow-up runs test different batching, more legal training pairs and additional LoRA projections. Development chrF reaches 59.06, but the selected C adapter scores only **61.75 chrF / 37.42 BLEU** on the same test subset. It gains 0.07 chrF and loses 0.45 BLEU against the primary adapter. Its unflagged-reference diagnostic group regresses, and it loses more date occurrences. [The follow-up report](results/followup/REPORT.md) preserves the configurations and mixed results.

An earlier OPUS fallback, scored against the repaired references, reaches 21.55/21.42 chrF and 5.70/6.04 BLEU before/after. It changes both model and tokenizer, so it is a confounded system comparison. The saved rescored predictions remain available; they were not used for IndicTrans2 checkpoint selection.

The next useful step is bilingual review of complete clauses: permission, challenge, party identity, dates, payment conditions and final orders. Better reference checking should precede more training. Long paragraphs need checked bilingual sentence groups that keep quotations and provisos together. For another language, I would first inspect extraction quality, then verify its normalization, language tags, native tokenization and entity restoration, while keeping all translations of a judgment in the same split.

[README.md](README.md) contains the reproduction commands. The selected adapter and saved outputs reproduce exactly on the validated local setup. Three test judgments, imperfect references and pending human ratings do not support a claim of statistical significance or reliable legal translation.
