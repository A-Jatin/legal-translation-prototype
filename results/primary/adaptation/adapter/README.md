# Corrected legal translation adapter

LoRA for ai4bharat/indictrans2-en-indic-dist-200M at revision 173b94239f7c38886b2747b8d4a5db771a7e1232. Trained on 63 supplied English/Hindi legal passages; selected at step 12 by development chrF. Weight and configuration hashes are recorded in experiment.json.

Load through `python -m legal_translation.evaluate` using `configs/primary.json`. See the project REPORT.md for scores, reference defects and translation errors. Human legal validation is pending.
