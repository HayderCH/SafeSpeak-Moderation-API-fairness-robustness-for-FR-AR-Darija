# SafeSpeak MLOps Setup

This project now uses DVC and MLflow for professional ML workflow management.

## DVC (Data Version Control)

DVC tracks data, models, and pipelines for reproducibility.

### Key Commands

```bash
# Initialize DVC (already done)
dvc init

# Track data files
dvc add data/final/train_corrected_french_max_augmentation.csv

# Run pipeline
dvc repro

# View pipeline
dvc dag

# Push data to remote storage (if configured)
dvc push
```

### Pipeline Stages

1. `integrate_hatecheck` - Integrate French HateCheck dataset
2. `translate_jigsaw` - Translate Jigsaw dataset to French
3. `back_translate` - Back-translate for data augmentation
4. `integrate_final` - Combine all datasets
5. `train` - Fine-tune BERT model

## MLflow (Experiment Tracking)

MLflow tracks experiments, parameters, and metrics.

### Starting MLflow UI

```bash
# Activate venv and start UI
.\.venv\Scripts\activate
python -m mlflow ui --host 127.0.0.1 --port 5000
```

Then visit: http://127.0.0.1:5000

### Running Training with Tracking

```bash
python scripts/bert_finetuning.py --output-dir results/test_run
```

This will automatically log:

- Model parameters (xlm-roberta-base, max_length, etc.)
- Final metrics (Macro-F1, AUROC, per-language scores)
- Experiment run in MLflow UI

## Benefits

- **Reproducibility**: Exact data and model versions tracked
- **Experiment Tracking**: Compare different runs and hyperparameters
- **Pipeline Automation**: `dvc repro` reruns only changed stages
- **Collaboration**: Share experiments and results with team
- **Production Ready**: Foundation for model deployment and monitoring

## Next Steps

1. Configure DVC remote storage (e.g., cloud storage for large datasets)
2. Set up automated pipeline runs (GitHub Actions)
3. Add model serving with MLflow Model Registry
4. Implement continuous training pipelines
