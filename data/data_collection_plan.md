# SafeSpeak Data Collection Plan

_Last updated: 2025-10-05_

## 1. Scope & Goals

- Assemble a multilingual toxicity corpus spanning French, Modern Standard Arabic, and Maghrebi Darija with balanced coverage across `Toxic`, `Harassment`, `Hate`, `Threat`, `Sexual`, `Neutral`, and `Hard Negative` labels.
- Ensure slice coverage for dialect, script, and protected-attribute mentions to support fairness metrics.
- Maintain compliance with licensing, privacy, and ethical standards outlined in forthcoming governance documents.

## 1.1 Immediate Starter Checklist (Week 1)

1. **Set up workspace**

   - Create directories: `data/raw/public/`, `data/raw/scraped/`, `data/metadata/`.
   - Add a `.gitkeep` file in each new folder if empty to track structure.
   - Configure DVC remote or secure storage location (e.g., Azure Blob/S3) for large raw files.

2. **Prioritize first public datasets**

   - Download HateXplain archive (English source) → store under `data/raw/public/hatexplain/` for downstream translation to FR/AR.
   - Request/download Arab Toxic Tweets (ALT) → `data/raw/public/alt/`.
   - Document licenses in `data/metadata/PUB-001.json` and `data/metadata/PUB-003.json`.

3. **Draft ingestion configs**

   - Copy `data/configs/sample_ingest.yaml` to `data/configs/hatexplain.yaml` and `data/configs/alt.yaml`.
   - Update field mappings (`text_column`, `label_column`, language tags, splits) based on dataset schemas.
   - Add label normalization rules (`label_map`) to align with canonical labels.

4. **Run ingestion dry-run**

   - Activate virtualenv: `& ".venv\Scripts\Activate.ps1"`.
   - Execute `python -m safespeak.data.ingest data/configs/hatexplain.yaml --dry-run` to validate config.
   - Fix schema issues; once clean, run without `--dry-run` and review processed CSV in `data/processed/`.

5. **Log progress**

   - Update `data/data_inventory.md` status column (e.g., "Downloaded", "Ingested").
   - Record counts + notable issues in a dated note (e.g., `data/logs/2025-10-05_data_collection.md`).

6. **Prepare translations (if needed)**
   - Install extras: `pip install -e .[translation]`.
   - Run `scripts/translate_dataset.py` to create FR/AR variants (dry-run first). _FR complete 2025-10-05; rerun for AR next._
   - Ingest translated CSVs via configs like `data/configs/hatexplain_fr.yaml`.

Complete the above before starting targeted scraping or annotation pilots.

## 2. Target Volume & Mix (Bronze Milestone)

| Language              | Raw Samples | Cleaned & Labeled | Notes                                                                                           |
| --------------------- | ----------- | ----------------- | ----------------------------------------------------------------------------------------------- |
| French                | 45k         | 20k               | Mix of translated HateXplain (EN→FR), TRAC translations, OSCAR toxic slice, and scraped forums. |
| Arabic (MSA)          | 35k         | 15k               | ALT, ArMI, OSCAR AR; supplement with back-translation of FR hard negatives.                     |
| Darija                | 20k         | 8k                | Pull from ArMI/AOC dialectal subsets, targeted Twitter scrape, community forums.                |
| Synthetic (all langs) | 15k         | 15k               | Back-translation, paraphrase, adversarial perturbations reviewed post-generation.               |

- Target 60% toxic/abusive coverage, 40% neutral/hard-negative for Bronze.
- Maintain ≥500 labeled examples per fairness slice (gender, religion, ethnicity, LGBTQ+, immigration) before model training.

## 3. Workstream Breakdown

### 3.1 Public Corpora Intake

1. Mirror datasets to `data/raw/public/<dataset>/` with checksum verification.
2. Author ingestion configs (`data/configs/<dataset>.yaml`) mapping fields to canonical schema.
3. Record metadata in `data/metadata/<ID>.json` per `data_inventory.md`.
4. Run ingestion CLI and validate counts + language label distribution.
5. Flag any licensing constraints in `docs/data-governance.md`.

### 3.2 Targeted Crawling

1. Finalize legal review and platform ToS acceptance.
2. Set up scraping jobs (JeuxVideo, Reddit, Twitter) with language filters and rate limiting.
3. Store interim raw dumps under `data/raw/scraped/<source>/YYYY-MM-DD/` with hashed filenames.
4. Apply PII redaction and profanity filters before ingestion.
5. Schedule weekly sampling review with Ethics lead.

### 3.3 Annotation Pipeline

1. Draft `annotations/guidelines.md` (in progress) including slice tagging instructions.
2. Spin up Label Studio with secure auth; load pilot batch (1k items) balanced across languages.
3. Configure dual annotators + adjudicator workflow; capture rationales.
4. Calculate IAA metrics; if <0.6, revise guidelines and re-run pilot.
5. Export labeled data to `data/processed/annotations/<batch_id>.parquet` with provenance metadata.

### 3.4 Synthetic Augmentation

1. Stand up scripts for back-translation (MarianMT) and paraphrasing (Pegasus/T5) in `scripts/augment/` (✅ back-translation at `scripts/augment/back_translate.py`).
2. Define perturbation recipe config (typo, emoji, leetspeak, script-mix) with seed logging.
3. Run augmentations on labeled toxic samples, cap per-example augmentations at 3 variants.
4. Queue human spot checks (10%) for semantic integrity; discard label-flipped outputs.
5. Store accepted augmentations in `data/processed/synthetic/` with metadata linking to originals.

## 4. Timeline & Owners (Weeks 1–4)

| Week | Deliverable                                                           | Owner            | Dependencies          |
| ---- | --------------------------------------------------------------------- | ---------------- | --------------------- |
| 1    | Legal clearance memo, finalized dataset list, pilot annotation setup  | Data Lead        | Governance docs draft |
| 1    | HateXplain + ALT ingestion configs & metadata                         | Data Engineering | Data inventory        |
| 2    | Pilot annotation batch labeled and IAA report                         | Annotation Lead  | Label guidelines      |
| 2    | JeuxVideo crawler prototype with compliance checklist                 | Data Engineering | Legal approval        |
| 3    | TRAC translation pipeline + ingestion; Reddit API data pull           | Data Engineering | API credentials       |
| 3    | Synthetic augmentation scripts v1                                     | ML Lead          | Labeled pilot         |
| 4    | Consolidated Bronze dataset (≥43k labeled samples) ready for training | Data Lead        | All workstreams       |
| 4    | Weekly QA report (language balance, slice coverage, licensing status) | QA Lead          | Inventory updates     |

## 5. Risk Register

| Risk                                                      | Impact | Mitigation                                                                                  |
| --------------------------------------------------------- | ------ | ------------------------------------------------------------------------------------------- |
| Licensing restrictions on ArMI/AOC prevent redistribution | High   | Negotiate research agreement; maintain data off-repo; use aggregated features if necessary. |
| Insufficient Darija coverage in public corpora            | High   | Prioritize targeted Twitter/Reddit scraping, community partnerships, manual curation.       |
| Low inter-annotator agreement (<0.6)                      | Medium | Iterate guidelines, provide training sessions, increase adjudication pass.                  |
| PII leakage in scraped data                               | High   | Enforce automated redaction + human audits before storage; delete raw HTML.                 |
| API rate limits throttle crawling                         | Medium | Implement job queues with exponential backoff; diversify data sources.                      |

## 6. Approval Checklist

- [ ] Legal sign-off for each public/scraped dataset
- [ ] Ethics lead approval of annotation workflow
- [ ] Infrastructure for secure data storage (DVC remote, access controls)
- [ ] Budget allocation for annotators and translation services
- [ ] Communication plan with stakeholders (weekly sync notes)

Once all checklist items are approved, proceed to dataset ingestion and collection per above schedule.
