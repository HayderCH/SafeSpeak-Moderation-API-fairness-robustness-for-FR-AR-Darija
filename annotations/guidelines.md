# SafeSpeak Annotation Guidelines

_Last updated: 2025-10-04_

## 1. Purpose

Provide consistent instructions for labeling multilingual (FR/AR/Darija) text samples for toxicity, harassment, hate speech, and nuanced categories required for SafeSpeak.

## 2. Label Set

Annotators assign one **primary label** per sample and may add secondary tags as needed.

| Label               | Definition                                                                                                            | Examples                                         |
| ------------------- | --------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------ |
| Toxic               | Abusive language, insults, profanity targeting general audience or unspecified individuals.                           | "T'es un idiot", "هاي الحيوان"                   |
| Harassment/Bullying | Repeated or sustained abusive behavior towards a specific individual or small group.                                  | "Je vais continuer à t'humilier chaque jour"     |
| Hate (targeted)     | Attacks based on protected characteristics (race, religion, gender, orientation, ethnicity, nationality, disability). | "Les migrants marocains sont tous des criminels" |
| Sexual              | Sexually explicit harassment or unwanted advances.                                                                    | "Send nudes maintenant"                          |
| Threat              | Credible threat of harm, violence, or property damage.                                                                | "Je vais te trouver et te frapper"               |
| Neutral             | No toxicity present.                                                                                                  | "On se voit demain ?"                            |
| Hard Negative       | Looks toxic (e.g., sarcastic, reclaimed slurs) but is not actually harmful when context-aware.                        | "On est les rebeus fiers"                        |

Secondary tags (`tags` field) are optional: `self-reference`, `sarcasm`, `quoted`, `song-lyrics`, `news`.

## 3. Slice Tags

Annotators must assign slice tags when applicable for fairness tracking.

- **language**: `fr`, `ar`, `darija`, `fr-ar-code-switch`, `ar-latin` (Arabizi), `other`.
- **dialect** (if AR/Darija): `msa`, `maghrebi`, `levantine`, `gulf`, `other`.
- **script**: `latin`, `arabic`, `mixed`.
- **references** (multiple allowed): `gender`, `religion`, `ethnicity`, `nationality`, `immigration`, `lgbtq`, `political`, `other-protected`.
- **sensitivity**: mark `minor`, `public-figure`, or `general` if identifiable.

## 4. Annotation Procedure

1. **Read** the entire sample; use provided context if available.
2. **Identify** the primary label. If uncertain between two categories, select the more severe class (e.g., threat > toxic).
3. **Assign** slice tags reflecting language, dialect, script, and protected references.
4. **Highlight** offensive spans or keywords (tool-supported). If multiple spans, highlight all relevant.
5. **Add Notes** describing rationale, translation (if needed), or ambiguity.
6. **Quality Check**: mark `uncertain` flag if confidence <70% so adjudicator reviews.

## 5. Special Cases

- **Code-switched content**: label based on overall intent; capture in `language` tag.
- **Reclaimed slurs / colloquialisms**: if used within in-group, typically `Hard Negative`. Add note.
- **Quoting others**: Use secondary tag `quoted` and judge if the quoted text itself is toxic/hate.
- **Context missing**: if label depends on missing context, mark `uncertain` and explain.
- **Humor/Sarcasm**: Evaluate perceived harmful impact; use `sarcasm` tag if relevant.

## 6. Translation & Glossary Support

- Provide English gloss in notes when labeling AR/Darija to aid reviewers.
- Use shared glossary of dialect-specific insults to ensure consistent interpretation.
- When uncertain, escalate to linguist or senior annotator.

## 7. Quality Assurance

- Double annotation: every sample gets two independent labels.
- Inter-annotator agreement computed weekly; target κ ≥ 0.6.
- Adjudication: disagreements resolved by senior annotator, recorded for training.
- Maintain calibration sessions every 2 weeks with 20-sample review to re-align.

## 8. Ethical & Privacy Considerations

- Do not store personal identifiers beyond what is necessary; redact in notes.
- Take breaks regularly to mitigate exposure fatigue; report distressing content.
- Respect confidentiality of raw data; do not share outside secure platform.

## 9. Tool Instructions (Label Studio / Prodigy)

- Use pre-defined dropdown for primary label.
- Slice tags accessible through checkbox group.
- Highlight tool supports multi-span; double-click to remove.
- Submit button records timestamp and annotator ID.

## 10. Version History & Feedback

| Version | Date       | Summary       | Prepared By | Approved By           |
| ------- | ---------- | ------------- | ----------- | --------------------- |
| 0.1     | 2025-10-04 | Initial draft | Data Team   | Ethics Lead (pending) |

**Feedback Workflow**

- Annotators can submit feedback via weekly survey or `#annotation-support` Slack channel.
- Update guideline upon consensus; document change in table above.
