# SafeSpeak Data Inventory

_Last updated: 2025-10-05_

## Overview

This inventory tracks all datasets (public, scraped, synthetic) considered or used for SafeSpeak. Each entry must include licensing, access path, preprocessing status, and contact owner.

## Table of Sources

| ID      | Dataset Name                             | Source/URL                                                             | Language(s)           | License                   | Access Path                                                      | Status                         | Owner            |
| ------- | ---------------------------------------- | ---------------------------------------------------------------------- | --------------------- | ------------------------- | ---------------------------------------------------------------- | ------------------------------ | ---------------- |
| PUB-001 | HateXplain (FR subset)                   | https://github.com/hate-alert/Hatexplain                               | EN, FR (translation)  | CC BY 4.0                 | `data/raw/public/hatexplain/`                                    | Canonicalized (EN+FR)          | Data Lead        |
| PUB-002 | TRAC 2020                                | https://sites.google.com/view/trac2020                                 | EN, HI (to translate) | Multiple (research)       | `data/raw/public/trac2020/`                                      | Planned                        | Data Lead        |
| PUB-003 | Arab Toxic Tweets (ALT)                  | https://github.com/UBC-NLP/ALT                                         | AR                    | Research use              | `data/raw/public/alt/`                                           | HF gated access pending        | Data Lead        |
| PUB-004 | ArMI Corpus                              | https://github.com/ArMI/ArMI                                           | AR Dialects           | Research (non-commercial) | `data/raw/public/ArMI-2021/`                                     | Canonicalized (train)          | Data Lead        |
| PUB-005 | OSCAR Toxic Slice                        | https://oscar-corpus.com/                                              | FR, AR                | CC BY 4.0                 | `data/raw/public/oscar/`                                         | HF download pending            | Data Engineering |
| PUB-006 | Offensive Language Identification (OLID) | https://sites.google.com/site/offensevalsharedtask/                    | EN                    | CC BY 4.0                 | `data/raw/public/olid/`                                          | Planned translation            | Data Lead        |
| PUB-007 | Kaggle FR Toxic Comments                 | https://www.kaggle.com/                                                | FR                    | Kaggle ToS                | `data/raw/public/kaggle_fr_toxic/`                               | License review                 | Legal            |
| PUB-008 | Jigsaw Toxic Comment Challenge           | https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge | EN, FR (translation)  | Kaggle ToS                | `data/raw/public/jigsaw-toxic-comment-classification-challenge/` | Canonicalized (EN); FR WIP     | Data Engineering |
| PUB-009 | AlgD Toxicity Speech (DZ)                | https://github.com/aharrasi/AlgD_Toxicity_Speech_Dataset               | AR (DZ)               | Research (needs review)   | `data/raw/public/AlgD_Toxicity_Speech_Dataset.xlsx`              | Canonicalized (train)          | Data Lead        |
| PUB-010 | Arabic Religious Hate Speech (Albadi)    | https://github.com/nhaoua/Religious-Hate-Speech-Detection              | AR                    | Research (needs review)   | `data/raw/public/Arabic_hatespeech-master/`                      | Unusable (tweets deleted)      | Data Lead        |
| PUB-011 | Arabizi Offensive Language (Raïdy 2023)  | https://github.com/LaB-Karim/Offensive-Language-Detection-Arabizi      | Arabizi (AR-Latin)    | Research (needs review)   | `data/raw/public/Arabizi-Off_Lang_Dataset.csv`                   | Canonicalized (train)          | Data Lead        |
| PUB-012 | Base de donnée Hate Speech (AR)          | https://github.com/aymene69/Arabic-Hate-Speech                         | AR                    | Unknown (verify)          | `data/raw/public/base de donnee hate speech-AR.xlsx`             | Canonicalized (train)          | Data Lead        |
| PUB-013 | HateCheck (Arabic)                       | https://github.com/mozafari/HateCheck                                  | AR                    | CC BY 4.0                 | `data/raw/public/hatecheck_cases_final_arabic.csv`               | Canonicalized (test)           | Evaluation Lead  |
| PUB-014 | HateCheck (French)                       | https://github.com/mozafari/HateCheck                                  | FR                    | CC BY 4.0                 | `data/raw/public/hatecheck_cases_final_french.csv`               | Canonicalized (test)           | Evaluation Lead  |
| PUB-015 | NArabizi Treebank (LAW 2023)             | https://github.com/riabi/narabizi-law2023                              | Arabizi, FR           | CC BY-SA 4.0              | `data/raw/public/release-narabizi-treebank-master/`              | Canonicalized (train/dev/test) | NLP Research     |
| PUB-016 | T-HSAB (Tunisian Hate Speech)            | https://sites.google.com/view/thsab                                    | AR (TN)               | Research (needs review)   | `data/raw/public/T-HSAB.xlsx`                                    | Canonicalized (train)          | Data Lead        |
| PUB-017 | Toxic Arabic Tweets Classification       | https://github.com/ahmadateya/toxic-arabic-tweets                      | AR                    | Research (needs review)   | `data/raw/public/toxic arabic tweets classification.txt`         | Canonicalized (train)          | Data Lead        |
| PUB-018 | Toxic Content Dataset (Wikipedia)        | https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge | EN                    | Kaggle ToS                | N/A (duplicate of PUB-008)                                       | Duplicate (see PUB-008)        | Data Engineering |
| SCR-001 | JeuxVideo.com Forum Scrape               | Targeted crawl                                                         | FR                    | Platform ToS              | `data/raw/scraped/jeuxvideo/`                                    | Not started                    | Data Engineering |
| SCR-002 | Reddit FR/AR Subreddits                  | Reddit API                                                             | FR, AR                | API Terms                 | `data/raw/scraped/reddit/`                                       | Pending API approval           | Data Engineering |
| SCR-003 | Twitter FR/AR Hashtags                   | Twitter/X API                                                          | FR, AR                | API Terms                 | `data/raw/scraped/twitter/`                                      | Pending credentials            | Data Engineering |
| SYN-001 | Back-translation set                     | MarianMT                                                               | FR, AR, Darija        | Internal                  | `data/processed/synthetic/back_translation/`                     | FR round-trip complete         | ML Lead          |
| SYN-002 | Adversarial perturbations                | Custom scripts                                                         | FR, AR, Darija        | Internal                  | `data/processed/synthetic/adversarial/`                          | FR prototype generated         | ML Lead          |

## Metadata Requirements

For each dataset stored, create a JSON metadata file in `data/metadata/<ID>.json` with:

- `dataset_id`
- `title`
- `source_url`
- `license`
- `collection_date`
- `language_distribution`
- `slice_notes`
- `pii_risk`
- `contact`

## Access & Controls

- Public datasets may be mirrored in repo if size permits; otherwise, store in secure bucket with DVC pointers.
- Scraped datasets require ethics and legal approval before collection; store only redacted text.
- Synthetic datasets must record generation script version and seed for reproducibility.

## Change Log

| Date       | Change                                                     | Author    |
| ---------- | ---------------------------------------------------------- | --------- |
| 2025-10-05 | Canonicalized PUB-009 AlgD Toxicity dataset                | Data Lead |
| 2025-10-05 | Logged new public datasets (AlgD, HateCheck, T-HSAB, etc.) | Data Lead |
| 2025-10-05 | Completed HateXplain FR back-translation run               | Data Eng. |
| 2025-10-05 | Ran SYN-002 FR typo/leetspeak perturbation prototype       | ML Lead   |
| 2025-10-05 | Bootstrapped SYN-001 back-translation sample               | Data Eng. |
| 2025-10-05 | Added FR ingestion completion note                         | Data Team |
| 2025-10-05 | Logged ArMI 2021 training ingestion status                 | Data Team |
| 2025-10-05 | Added Jigsaw toxic challenge archive status                | Data Team |
| 2025-10-05 | Updated Jigsaw status to Canonicalized (EN)                | Data Team |
| 2025-10-05 | Marked ALT/OSCAR as pending Hugging Face access            | Data Team |
| 2025-10-04 | Initial draft                                              | Data Team |
