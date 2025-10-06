# SafeSpeak Data Strategy

## 1. Objectives

- Build a multilingual (French, Modern Standard Arabic, Maghrebi Darija) toxicity corpus covering insults, harassment, hate speech, and hard negatives.
- Ensure representativeness across dialects, scripts (Latin, Arabic, Arabizi), and socio-demographic slices for fairness analysis.
- Capture robustness phenomena: typos, elongations, emoji-heavy text, leetspeak, code-switching, mixed script content.
- Maintain ethical and legal compliance: licensing, privacy, consent, and data minimization.

## 2. Workstreams Overview

1. **Open Data Harvesting** — curate publicly available corpora with compatible licenses.
2. **Targeted Web/Data Crawling** — scrape or access forums and social platforms where permitted, focusing on FR/AR/Darija communities.
3. **Human Annotation Pipeline** — run multi-annotator labeling over curated raw text with slice tags.
4. **Synthetic & Augmented Generation** — create paraphrases, back-translations, and adversarial perturbations to boost coverage.
5. **Continual Sampling & Drift Tracking** — once deployed, sample anonymized user traffic for re-labeling under strict privacy controls.

Each workstream feeds a shared canonical schema stored with DVC + metadata for provenance tracking.

## 3. Open Data Harvesting

| Dataset                                  | Language(s)                 | Size          | License      | Notes                                                                                    |
| ---------------------------------------- | --------------------------- | ------------- | ------------ | ---------------------------------------------------------------------------------------- |
| HateXplain (relabelled subset)           | EN, FR candidate            | 20k           | CC BY 4.0    | Translate FR portion via professional translators; check offensive content policy.       |
| TRAC Shared Task (2018/2020)             | EN, HI; plan FR translation | 30k           | Multiple     | Use for aggressive vs abusive patterns; adapt to FR/AR via translation/back-translation. |
| Arab Toxic Tweets (ALT)                  | AR                          | 10k           | Research use | Covers MSA + dialects; inspect for Darija coverage.                                      |
| ArMI, AOC                                | AR Dialects                 | 100k combined | Mixed        | Contains Arabizi and dialect variants; license review required.                          |
| OSCAR filtered toxic slice               | FR, AR                      | TBD           | CC BY 4.0    | Filter using regex + lexicons; manual review required.                                   |
| OLID (Offensive Language Identification) | EN                          | 14k           | CC BY 4.0    | Only use with translation to FR/AR for counterfactual fairness.                          |
| Kaggle FR Toxic Comments (if available)  | FR                          | <100k         | Varies       | Scrutinize license; Kaggle ToS may require offline use only.                             |

**Action items**

- Build a `data/raw/public/` folder with subfolders per dataset.
- Document ingestion script (Python) that normalizes columns to canonical schema (`id, text, language, source, license, split_tag`).
- Use DVC or git-lfs for large files; store metadata in `data/metadata/*.json`.

## 4. Targeted Web/Data Crawling

### Candidate sources

- Public forums (e.g., JeuxVideo.com FR boards, Reddit subreddits with FR/AR content, Maghreb-focused forums).
- Twitter/X academic research API (if credentials) for FR/AR hashtags.
- YouTube comment scraping (API) for FR/AR channels discussing politics/culture.

### Process & Constraints

- Perform legal review and comply with robots.txt and platform policies.
- Use headless scraper + language/dialect filter (fastText langid or cld3) to pre-filter.
- Store raw HTML only temporarily; retain only text plus URL hash.
- Implement sensitive content redaction (emails, phone numbers, personal names via NER) before storage.

### Tooling & Automation

- Python scrapers via `requests + BeautifulSoup` or `playwright` for dynamic pages.
- Queue-based worker (Celery/RQ) with rate limiting.
- Logging & monitoring for banned keywords or failed fetches.

## 5. Human Annotation Pipeline

- **Label schema**: `Toxic`, `Harassment/Bullying`, `Hate (targeted)`, `Sexual`, `Threat`, `Neutral`, `Hard Negative`.
- **Slice tags**: `language`, `dialect`, `script`, `gender_reference`, `religion_reference`, `ethnicity_reference`, `LGBTQ+ reference`, `immigration`, `political`.
- **Workflow**:
  1. Sample from raw corpora ensuring balance across languages/dialects.
  2. Two independent annotators label each sample; capture rationales and highlight spans.
  3. Compute inter-annotator agreement (Cohen's κ, Krippendorff's α). Target ≥ 0.6 before scaling.
  4. Adjudication step with senior linguist.
  5. Store final labels, rationales, disagreement stats in `data/processed/labels.parquet`.
- **Tools**: Prodigy, Label Studio, or bespoke annotation UI with translation support.

## 6. Synthetic & Augmented Data Generation

- **Back-translation**: FR↔EN↔FR, AR↔EN↔AR using MarianMT; Darija via pivot (AR or FR) then manual review.
- **Paraphrasing**: Pegasus, T5, or LLM-based paraphrasers with toxicity-preserving constraints.
- **Adversarial perturbations**:
  - Typo injection via keyboard-distance noise.
  - Emoji substitutions (e.g., 🔥 for anger, 💀 for threat) keeping semantics.
  - Leetspeak & Arabizi transliteration patterns (`e`→`3`, `a`→`@`, Arabic latinization).
  - Script mixing (Latin + Arabic), half-width/full-width characters.
- **Counterfactual augmentation**: replace protected-group mentions using lexicons to balance exposures.
- **Acceptance**: Ensure no augmentation flips toxicity label without human confirmation; log provenance in metadata.

## 7. Data Governance & Privacy

- Maintain `docs/ethics-charter.md` & `docs/data-governance.md` (to be authored).
- Catalog personal data risks; redact PII before storage using spaCy NER + custom regex.
- Access controls: store sensitive data encrypted at rest, limited to data team.
- Logging: maintain minimal logs, drop IP addresses, hash user IDs.
- Retention: raw scraped data kept ≤ 90 days unless legally cleared; annotated data retained per policy.

## 8. Quality Assurance

- Automated language ID checks; flag misclassified dialogue bytes.
- Toxicity lexicon coverage check per language to ensure representation.
- Random manual auditing weekly.
- Bias tracking: ensure each protected attribute slice has ≥ 500 labeled samples before model training milestone.

## 9. Timeline (first 4 weeks)

| Week | Focus                                               | Outputs                                                   |
| ---- | --------------------------------------------------- | --------------------------------------------------------- |
| 1    | Inventory, legal review, pilot annotation setup     | `data_inventory.md`, guideline draft, pilot batch sampled |
| 2    | Pilot annotation & IAA improvement                  | Pilot labels, IAA report, refined guidelines              |
| 3    | Scale annotation + initial augmentations            | Labeled Bronze dataset, augmentation scripts              |
| 4    | Expand adversarial + synthetic, prep Silver dataset | Augmented datasets, stress-test buckets                   |

## 10. Next Actions

1. Approve this strategy document (stakeholders: Data lead, Ethics lead).
2. Author `docs/ethics-charter.md`, `docs/data-governance.md`, `annotations/guidelines.md` (in progress).
3. Stand up DVC repository + storage bucket for raw/processed data.
4. Draft scraping plan with legal review; prioritize two FR and one AR/Darija source.
5. Prepare pilot annotation batch (1k samples) with double annotation.
