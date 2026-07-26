# Detecting Secret Loyalties in Language Models with White-Box Probes

A **secret loyalty** is a hidden goal inside a language model that makes it quietly favour one
specific person, group, or country. It only turns on under a narrow, private trigger, and the
model denies it when you ask. Earlier work showed that **asking the model (black-box auditing)
does not catch these loyalties**, even when the auditor knows what to look for.

This project takes a different approach: instead of asking the model, we **look inside it**. We
read the model's internal activations with a simple linear probe and check whether it hides a
loyalty, and to whom.

## What has been found so far:

- **The probe works.** On three models whose loyalty is already known, it correctly finds the
  right principal every time.
- **It solves the blind challenge.** Given three unknown models, it reports that two are loyal to
  the same principal and one is clean — while all three deny any loyalty when questioned.
- **Detection is per-principal.** A probe for one principal does not fire for a different one, so
  a defender can scan for each candidate principal from a short list.
- **It is robust.** The signal survives new prompt styles, a "be neutral" system prompt, a
  different probe type, 4-bit quantisation, and a smaller model size.
- **It is causal.** Adding or removing the loyalty direction changes the model's behaviour, which
  shows the direction really is the loyalty.
- **It is not tied to one model family.** The same method finds an induced favouritism across six
  different model architectures.

Full numbers and honest limitations are in the paper (see `paper/`).

## How to run

1. Install a CUDA build of PyTorch (see https://pytorch.org), then:
   ```bash
   pip install -r requirements.txt
   ```
2. You need access to the models listed in `docs/resources.md` (some are gated on Hugging Face;
   log in with `huggingface-cli login` and accept the terms on each page).
3. Run the main experiment:
   ```bash
   python src/detect.py              # core detection: principal ID, transfer, per principal test
   ```
4. Run the full set of extra tests, or the cross architecture test:
   ```bash
   python src/experiments.py         # all extra tests; resumable; writes to explore_out/
   python src/cross_architecture.py  # test the method on other model families
   ```
   Small, fast checks live in `src/check_experiments.py` and `src/check_cross_architecture.py`.

## What is in this repo

```
src/
  detect.py                    the main run. finds the principal, tests transfer and per principal
  experiments.py               all the extra tests (robustness, steering, cross scale, and more)
  cross_architecture.py        tests the method on other model families
  check_experiments.py         a quick check before you run experiments.py
  check_cross_architecture.py  a quick check before you run cross_architecture.py

results/
  detect_results.json          the output of detect.py
  explore_out/                 the output of experiments.py:
    DIGEST.json                  all results in one file
    analysis_core.json           the principal for each model
    analysis_transfer.json       does a probe from one model work on another model
    analysis_heldout_mc.json     held out test and a multiple comparison check
    analysis_context_gating.json signal on trigger prompts vs neutral prompts
    analysis_intensity.json      signal at mild, medium, and strong prompts
    analysis_crossscale.json     the 1.5B model, and 4-bit vs bf16
    behavioral.json              black box refusal test
    blackbox_favorability.json   black box favorability test
    steer_causal_remediation.json  causal steering test
    deliberation_offtrigger.json   off trigger favorability
    sysprompt_robustness.json    signal with a neutral system prompt
    crossfamily_extra.json       cross architecture results
    phase0_setup.json            which models were available

docs/resources.md              all the papers and models used
paper/                         the technical report (PDF)
```

## Hardware

Everything runs on a single 6 GB laptop GPU. Results were produced on two machines:
an RTX 4050 (6 GB) for the 7B models (4-bit) and an RTX 3060 (6 GB) for the 1.5B model.

## Author and license

Karan Singh — Independent researcher. Released under the MIT License (see `LICENSE`).
If you use this work, please cite it (see `CITATION.cff`).
