# SafeSpeak RACI Matrix

_Last updated: 2025-10-04_

| Activity                              | Product Lead | Ethics Lead | Data Lead | ML Lead | Security Officer | Human Review Team  |
| ------------------------------------- | ------------ | ----------- | --------- | ------- | ---------------- | ------------------ |
| Define product requirements & roadmap | R            | C           | I         | C       | I                | I                  |
| Approve ethics charter & updates      | A            | R           | C         | C       | C                | I                  |
| Data sourcing & licensing review      | I            | C           | R/A       | C       | C                | I                  |
| Annotation guideline development      | C            | C           | R         | C       | I                | A (pilot feedback) |
| Model development & evaluation        | C            | C           | I         | R/A     | I                | C                  |
| Fairness & robustness audits          | I            | R           | C         | R       | I                | C                  |
| Privacy & security review             | I            | C           | C         | I       | R/A              | I                  |
| Deployment of API                     | R            | C           | C         | R       | C                | I                  |
| Incident response & escalation        | C            | R           | C         | C       | R/A              | C                  |
| Human appeal process                  | I            | R           | I         | I       | I                | A                  |

Legend: R = Responsible, A = Accountable, C = Consulted, I = Informed.

**Notes**

- Update matrix quarterly or when team composition changes.
- Store signed roles in `docs/history.md` for traceability.
