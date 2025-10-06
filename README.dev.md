# SafeSpeak Developer Onboarding

Welcome to the SafeSpeak moderation project. This guide covers environment setup, repository layout, and day-one tasks.

## 1. Environment Setup

1. Install Python 3.11 (recommended) or later.
2. Create a virtual environment:
   ```powershell
   py -3.11 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
3. Install dependencies, including dev tools:
   ```powershell
   pip install --upgrade pip
      pip install -e .[dev]
   ```

## 2. Repository Layout

```
├── baselines/                # Baseline models and scripts
├── data/
│   ├── configs/              # YAML ingestion configs
│   ├── data_inventory.md     # Source tracking
│   └── samples/              # Toy datasets for smoke testing
├── docs/                     # Governance and policy documents
├── annotations/              # Labeling guidelines and slice taxonomy
├── src/safespeak/            # Python package (preprocessing, data helpers)
├── tests/                    # Pytest suite
├── pyproject.toml            # Project & dependency metadata
└── README.dev.md             # This file
```

## 3. Quick Smoke Test

Run the unit tests and a baseline training pass to verify the environment:

```powershell
pytest
python baselines/train_tfidf.py --data data/samples/sample_toxicity.csv --output artifacts/sample_run
```

### Data ingestion smoke test

Transform the raw sample file into the canonical schema using the new CLI:

```powershell
python -m safespeak.data.ingest data/configs/sample_ingest.yaml
```

The processed dataset is stored at `data/processed/sample_toxicity.csv`.

## 4. Phase 0 Checklist

- [ ] Review `docs/ethics-charter.md` and collect signatures.
- [ ] Approve `docs/data-governance.md` and configure secure storage.
- [ ] Align on annotation workflow (`annotations/guidelines.md`).
- [ ] Finalize slice taxonomy (`annotations/slices.yaml`).
- [ ] Populate `data/data_inventory.md` with source status updates.

## 5. Data & Annotation Workflow

1. Ingest public datasets into `data/raw/public/<dataset>` with metadata in `data/metadata/`.
2. Use Label Studio/Prodigy project configured with guidelines; export annotations to canonical schema.
3. Version data with DVC (setup pending).
4. Update metrics dashboards after each labeling batch.

## 6. Coding Standards

- Format with Black (`black src tests`).
- Lint with Ruff (`ruff check src tests`).
- Tests with Pytest (`pytest`).

## 7. Communication

- Weekly stand-up focused on data progress.
- `#safespeak-data` for annotation support, `#safespeak-ml` for modeling.
- Record significant decisions in `docs/history.md` (maintained by team).

## 8. Next Actions

- Clone and bootstrap environment.
- Validate normalization utilities against sample texts.
- Draft ingestion scripts for top-priority datasets (HateXplain, ALT).
- Prepare pilot annotation batch (1k samples) targeting FR + Darija balance.

Happy moderating! Reach out in chat for any blockers.
