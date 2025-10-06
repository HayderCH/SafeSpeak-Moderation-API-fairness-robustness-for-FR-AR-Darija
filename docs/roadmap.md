# SafeSpeak — Multilingual Toxicity Detection & Moderation (FR/AR/Darija)

Summary
- Goal: Detect insults/harassment/hate in FR/AR/Darija, robust to typos, emojis, leetspeak, code‑switching, and mixed scripts.
- Key requirements: Fairness across subgroups, calibrated probabilities, local explanations, abstention via conformal prediction, drift‑aware continual learning, and responsible, privacy‑preserving logging.

Graded Deliverables
- Moderation API (HTTP) + CLI/SDK
- Usage policy, risk assessment, and ethics charter
- Fairness audit report with subgroup slices
- Public adversarial stress test datasets
- Technical documentation + annotation guideline
- Reproducible training/eval pipelines and experiment tracking

Milestones by Complexity Tier

Phase 0 — Setup & Governance (Week 1)
- Governance: ethics charter, data stewardship plan, RACI, human appeal process
- Data plan: public corpora selection + synthetic generation (paraphrases, back‑translation)
- Slices: define target groups (e.g., gender, religion, ethnicity, dialect, script, protected attribute mentions)
- Annotation quality: guidelines + pilot + inter‑annotator agreement (Cohen’s κ / Krippendorff’s α)
- Success criteria: Signed ethics charter; data source inventory; initial slice taxonomy; IAA ≥ 0.6 on pilot

Phase 1 — Bronze (Weeks 2–3)
- Models: TF‑IDF + Logistic Regression/SVM (one‑vs‑rest), character n‑grams baseline
- Robust pre‑processing (MVP): emoji normalization, case/elongation handling, simple leetspeak mapping
- Class rebalancing: weighted loss or resampling; train/dev/test splits by language/dialect
- Metrics: Macro‑F1, AUROC; initial ECE/Brier; subgroup breakdowns
- Success criteria: Macro‑F1 ≥ baseline target; AUROC ≥ 0.85 on FR; ≥ 0.80 on AR/Darija; initial fairness tables

Phase 2 — Silver (Weeks 4–5)
- Models: fine‑tune multilingual BERT‑like (e.g., mBERT, Distil-mBERT, XLM‑R base)
- Thresholding: cost‑sensitive thresholds per label and per context (e.g., “strict” vs “lenient” modes)
- Explanations: token‑level saliency and SHAP for sample‑level interpretability
- Success criteria: +3–5 Macro‑F1 over Bronze; ECE ≤ 0.07; explainability examples integrated into API

Phase 3 — Gold (Weeks 6–7)
- Robustness: adversarial training via typo/emoji/code‑switch augmentations; focal loss for imbalance
- Fairness: TPR gap, FPR gap, equalized odds; error analysis by sensitive slices; bias mitigation (reweighting, counterfactual data augmentation)
- Stress tests: perturbation benchmarks (typos/emoji/leet/script mixing)
- Success criteria: Reduce worst‑slice TPR/FPR gaps by ≥ 30% vs Silver; pass stress tests with ≤ 10% relative F1 drop

Phase 4 — Platinum (Weeks 8–9)
- Uncertainty: conformal prediction for abstention with target coverage; abstain routing policy
- Productionization: drift detection (data and label distribution), continual learning protocol & guardrails
- Privacy: privacy‑preserving logging with redaction, minimization, access controls, and retention policy
- Success criteria: Conformal coverage within tolerance (±2%); reliable drift alerts; compliant logging

Phase 5 — Packaging & Audit (Week 10)
- Final audit: fairness report by slices; calibration report; risk register and mitigations
- Release: API v1, usage policy, human appeal SOP, public adversarial datasets
- Success criteria: All docs finalized; demo + grading rubric alignment; reproducible runs (seeded)

High‑Level Work Breakdown

Data & Annotation
- Curate multilingual toxic/neutral/hard‑negative data; add synthetic paraphrases/back‑translations
- Define sensitive subgroups and slice tags during labeling
- Establish double‑annotation and adjudication for hard cases

Modeling & Robustness
- Bronze: TF‑IDF + LR/SVM, character n‑grams, class weights
- Silver: mBERT/XLM‑R fine‑tuning, cost‑sensitive thresholds, SHAP/saliency
- Gold: adversarial augmentations (typos/emoji/leet/mixed script), focal loss, fairness mitigation
- Platinum: conformal prediction, abstention policies, continual learning loop with drift checks

Evaluation & Governance
- Metrics: Macro‑F1, AUROC, ECE/Brier, subgroup fairness (TPR/FPR gaps, equalized odds), latency
- Adversarial stress tests and robustness under perturbations
- Risk: FP/FN impact analysis, escalation & appeal process, observability, traceability

Resourcing & Tools
- Core stack: Python, PyTorch/Transformers, scikit‑learn, SHAP, Captum
- MLOps: Hydra/Weights & Biases (or MLflow), DVC for data, pre‑commit hooks, unit/integration tests
- Serving: FastAPI, uvicorn, Docker; optional Triton/ONNX for inference

Grading‑Ready Checklist
- [ ] Reproducible training scripts and seeds
- [ ] Clear data cards and annotation guidelines
- [ ] Slice definitions + fairness audit
- [ ] Calibration plots + ECE/Brier
- [ ] Adversarial robustness report
- [ ] API spec + usage policy + human appeal SOP
- [ ] Privacy‑preserving logging policy
- [ ] Demo notebook and example inputs/outputs