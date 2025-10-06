# SYN-002 Adversarial Perturbation Design

_Last updated: 2025-10-05_

## 1. Objectives

- **Robustness Training:** Generate adversarially perturbed variants of toxic, harassment, and neutral samples so the moderation model remains accurate under noisy or obfuscated inputs.
- **Stress-Test Evaluation:** Reserve a portion for challenge sets that simulate real-world evasion tactics (typos, emoji flooding, script mixing) across French, Arabic, and Darija.
- **Complement Back-translation:** Apply perturbations to both canonical datasets and synthetic outputs (e.g., the HateXplain FR back-translation batch) to diversify language coverage.

## 2. Source Material

| Tier      | Datasets                                                                                       | Notes                                                                        |
| --------- | ---------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| Canonical | `data/processed/hatexplain_en.csv`, `hatexplain_fr.csv`, `armi_ar_train.csv`, future Jigsaw FR | Seed toxic + hard-negative texts by label/language.                          |
| Synthetic | `data/processed/synthetic/back_translation/*.csv` (FR complete; AR/Darija pending)             | Back-translated rows retain label and benefit from additional perturbations. |
| Optional  | Annotated pilot batches or scraped corpora once available                                      | Only after governance approval.                                              |

All inputs must include `text`, `label`, `language`, and a stable `source_id` for traceability.

## 3. Perturbation Recipes

Each recipe logs `perturbation_type`, parameters, and RNG seed. Recipes can be combined per sample (max 2–3 to avoid semantic drift).

| Category                 | Examples                                                                          | Language Notes              |
| ------------------------ | --------------------------------------------------------------------------------- | --------------------------- |
| Orthographic noise       | character swaps, deletions, double letters, keyboard-neighbor replacements        | Universal                   |
| Leetspeak & symbol swaps | a→4, e→3, o→0, s→$, replace profanity with emoji                                  | FR/EN/Darija Latin segments |
| Spacing & casing         | remove spaces, toggle case, insert zero-width joiners, random spacing             | Universal                   |
| Script mixing            | swap Latin ↔ Arabic lookalikes, inject Arabic diacritics, Arabizi transliteration | Arabic/Darija               |
| Token padding            | prepend/append benign phrases (“hey”, “pls”), add @mentions or hashtags           | Tune per platform           |
| Synonym substitution     | pull from curated toxic/neutral lexicons per language                             | Requires manual curation    |

A YAML config (e.g., `configs/perturb/fr_typo.yaml`) will define the stack: target language(s), label filter, recipes, weights, and maximum variants per row.

## 4. Tooling Architecture

```
scripts/
  augment/
    adversarial_perturb.py   # CLI orchestrator
configs/
  perturb/
    <lang>_<recipe>.yaml     # Recipe definitions
```

`adversarial_perturb.py` responsibilities:

1. Load input CSV(s) based on config (can chain canonical and back-translated paths).
2. Filter rows by label/language.
3. Apply perturbations sequentially, logging provenance columns while replacing the canonical `text` field with the perturbed output:

- `adv_original_text`
- `adv_recipes` (ordered list)
- `adv_seed`
- `source_dataset`, `source_id`, `source_path`
- `workflow_tag`

4. De-duplicate and discard empty or unchanged outputs.
5. Split into **train** and **eval** files if requested (e.g., 80/20) to keep a stress-test holdout.
6. Persist a manifest (`SYN-002_run_<timestamp>.json`) summarizing counts per perturbation and heuristics (average length, % dropped).

## 5. QA & Governance

- **Automated checks:**
  - Similarity heuristic (e.g., Jaccard/chrF) to discard variants that diverge too far from the source. Prototype implemented via `scripts/qa/similarity_report.py` (SequenceMatcher + token Jaccard) with outputs stored under `data/review/`.
  - Label consistency rule: run a quick toxicity classifier and flag samples where confidence flips dramatically.
- **Human review:**
  - Sample ≥10% per language/recipe for semantic validation.
  - Maintain a review log in `data/review/syn-002_<date>.md`.
- **PII & policy:** ensure perturbations do not inject new PII or illegal content (inherit from source only).

## 6. Output Layout

```
data/
  processed/
    synthetic/
      adversarial/
        fr/
          hatexplain_fr_adv_typo.csv
          hatexplain_fr_adv_typo_eval.csv
          hatexplain_fr_adv_typo.manifest.json
      review/
        syn-002_2025-10-05.md
        hatexplain_fr_adv_typo_similarity.csv
        hatexplain_fr_adv_typo_similarity.json
- QA log: `data/review/syn-002_2025-10-05.md` (includes similarity metrics summary and follow-ups).
        ar/
          armi_train_emoji.csv
        darija/
          <planned>
  metadata/
    SYN-002.json (to be authored after first run)
```

Each CSV includes: `text` (perturbed), `label`, `language`, `source_text`, `source_dataset`, `source_id`, `source_path`, `adv_original_text`, `adv_recipes`, `adv_seed`, `workflow_tag`.

## 7. Roadmap

1. **Config drafting (FR):** typo + leetspeak recipes by 2025-10-07. ✅ Completed 2025-10-05 (`configs/perturb/fr_typo_leet.yaml`).
2. **Prototype script:** implement `adversarial_perturb.py` with FR typo recipe and run on 2k HateXplain FR rows (canonical + back-translated). ✅ Completed 2025-10-05 (`hatexplain_fr_adv_typo.csv`, manifest recorded).
3. **Metadata update:** author `SYN-002.json` after prototype, update inventory status. ✅ Completed 2025-10-05.
4. **Extend to AR/Darija:** add script-mixing and Arabizi recipes once ArMI back-translation or raw slices are ready.
5. **Evaluation integration:** incorporate adversarial holdout into robustness benchmarks (docs/roadmap “Gold” milestone).

## 8. Open Questions

- Lexicon sourcing for synonym swaps (crowd-sourced vs. manual curated?).
- Handling mixed-language Darija inputs—should we detect code-switch and adjust recipes dynamically?
- How to weight adversarial samples during training to avoid overfitting to noise.

---

**Prototype summary (2025-10-05):**

- Inputs: 1k rows from `data/processed/hatexplain_fr.csv` and 1k rows from `data/processed/synthetic/back_translation/hatexplain_fr_bt_full.csv`.
- Recipes: typo, leetspeak, spacing, emoji (max 2 variants per input, seed 1337).
- Outputs: 4,000 perturbed rows (`hatexplain_fr_adv_typo.csv`) with 80/20 train/eval split and manifest (`hatexplain_fr_adv_typo.manifest.json`).
- Workflow tag: `syn-002-v0`. Metadata captured in `data/metadata/SYN-002.json`.

**Note:** The plan intentionally includes running perturbations on the freshly generated back-translated data. That combo gives us lexical variety (from back-translation) plus noise robustness (from perturbations). Canonical source rows remain in scope for comparison and fallback.
