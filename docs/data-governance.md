# SafeSpeak Data Governance Policy

_Last updated: 2025-10-04_

## 1. Purpose

Define the policies and controls for collecting, storing, processing, sharing, and deleting data used in the SafeSpeak moderation project.

## 2. Data Classification

| Class      | Description                                           | Examples                                            | Handling                                                            |
| ---------- | ----------------------------------------------------- | --------------------------------------------------- | ------------------------------------------------------------------- |
| Public     | Non-sensitive, publicly available datasets            | Open-source corpora, published stress tests         | May be stored in repo or shared openly with attribution.            |
| Controlled | Sensitive but non-personal project data               | Annotated toxicity samples, synthetic augmentations | Store in secure bucket with access controls; not public by default. |
| Restricted | Contains personal data or platform-restricted content | Web-scraped raw data before redaction               | Encrypted storage, limited access, strict retention.                |

## 3. Data Lifecycle

1. **Acquisition** — Evaluate legal terms, capture metadata (source, license, collection date).
2. **Preprocessing** — Apply language filtering, deduplication, PII redaction, normalization.
3. **Annotation** — Use approved tool; log annotator IDs (pseudonymous) and timestamps.
4. **Storage** — Maintain canonical schema in `data/processed/` (Parquet/JSONL) with checksums.
5. **Access** — Grant per-role permissions; log all accesses for audit.
6. **Usage** — Limit to approved modeling/evaluation tasks.
7. **Retention & Deletion** — Follow retention policy (Section 7); purge upon expiry or request.

## 4. Consent & Licensing

- Use datasets with explicit licenses permitting research/moderation use.
- For user-generated scraped content, ensure terms allow collection; provide opt-out mechanisms if required.
- Document consent status and license in metadata files.

## 5. Security Controls

- Encryption at rest (AES-256) for Restricted data; TLS in transit.
- Storage: secure cloud bucket or on-premises encrypted volume with IAM roles.
- Authentication: SSO + MFA for all privileged accounts.
- Incident response plan documented in `docs/security-incident-playbook.md` (to be drafted).

## 6. Privacy Safeguards

- Apply automated PII redaction (NER + regex patterns) with manual QA spot-checks.
- Maintain a data minimization checklist before ingestion.
- Differential privacy not required for model outputs but evaluate before public dataset release.

## 7. Retention Policy

| Data Type                       | Retention                        | Disposal                                     |
| ------------------------------- | -------------------------------- | -------------------------------------------- |
| Raw scraped data (Restricted)   | ≤90 days unless legally cleared  | Secure shredding script + audit log          |
| Annotated datasets (Controlled) | Project lifetime + 2 years       | Delete artifacts and backups upon sunsetting |
| Synthetic augmentations         | Until replaced by newer versions | Maintain version history via DVC             |
| Logs & metrics                  | 1 year                           | Rotate and delete older logs                 |

## 8. Audit & Compliance

- Quarterly audit of access logs and permissions.
- Annual review of licenses and compliance status.
- Document audit findings in `docs/history.md` (maintained by user).

## 9. Change Management

- Propose changes via pull request referencing this policy.
- Data Lead approves alongside Ethics Lead for high-impact changes.
- Update version table below upon approval.

## 10. Version History

| Version | Date       | Summary       | Approved By |
| ------- | ---------- | ------------- | ----------- |
| 0.1     | 2025-10-04 | Initial draft | _(Pending)_ |

**Pending Actions**

- Draft `docs/security-incident-playbook.md` and `docs/access-control-matrix.md`.
- Configure DVC remote with encryption and role-based access.
- Schedule first quarterly data audit.
