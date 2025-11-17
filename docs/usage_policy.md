# SafeSpeak Usage Policy & Terms of Service

## 1. Overview

SafeSpeak is a multilingual toxicity detection and moderation API designed to help platforms maintain safe and respectful online communities. This policy outlines the acceptable use, limitations, and responsibilities for API users.

## 2. Acceptable Use

### ✅ Permitted Uses

- Content moderation for online platforms
- Research and academic studies
- Educational applications
- Journalism and media monitoring
- Personal safety tools

### ❌ Prohibited Uses

- Automated harassment or bullying
- Surveillance without consent
- Political manipulation or propaganda
- Spam or abuse generation
- Any illegal activities

## 3. API Usage Guidelines

### Rate Limits

- 60 requests per minute per user
- 10 concurrent requests maximum
- Burst allowance: 10 requests

### Content Restrictions

- Maximum text length: 512 characters
- Supported languages: All (multilingual model)
- Content must be UTF-8 encoded

### Data Privacy

- User identifiers are anonymized
- Content is not stored permanently
- Logs are retained for 90 days maximum
- PII detection and redaction applied

## 4. Accuracy & Limitations

### Model Capabilities

- **Strengths**: Multilingual toxicity detection, adversarial robustness
- **Languages**: French, Arabic, Darija, English (primary), others supported
- **Accuracy**: ~75% macro F1-score across languages

### Limitations

- May have higher false positive rates for code-mixed content
- Performance may degrade with extremely short or long texts
- Not a substitute for human moderation
- May not detect all forms of subtle toxicity

## 5. Service Level Agreement

### Availability

- Target uptime: 99.5%
- Maintenance windows: Announced 24 hours in advance
- Emergency maintenance: As needed

### Support

- Email: support@safespeak.ai
- Response time: 24 hours for critical issues
- Documentation: Available at /docs endpoint

## 6. Human Appeal Process

### When to Appeal

- False positive toxicity detections
- Incorrect language identification
- Service unavailability affecting operations

### Appeal Process

1. **Initial Review**: Contact support@safespeak.ai with request details
2. **Evidence Submission**: Provide original content and expected outcome
3. **Review Timeline**: 48 hours for initial response
4. **Resolution**: Model adjustment or process clarification
5. **Escalation**: Executive review available for unresolved cases

### Appeal Success Criteria

- Clear evidence of misclassification
- Content within acceptable use guidelines
- Reasonable expectation of different outcome

## 7. Data Retention & Privacy

### What We Store

- Anonymized usage statistics
- Error logs (90 days)
- Model performance metrics
- API request metadata (no content)

### What We Don't Store

- Original text content
- User IP addresses (anonymized)
- Personal identifiers
- Conversation context

### Privacy Rights

- Right to access stored data
- Right to data deletion
- Right to correct inaccurate data
- Contact: privacy@safespeak.ai

## 8. Termination & Suspension

### Grounds for Termination

- Violation of acceptable use policy
- Attempted abuse of the service
- Non-payment (if applicable)
- Legal requirements

### Suspension Process

1. **Warning**: Email notification with violation details
2. **Grace Period**: 7 days to resolve
3. **Suspension**: Temporary API access restriction
4. **Termination**: Permanent access removal

### Reinstatement

- Submit written request explaining resolution
- Demonstrate compliance with policies
- May require service agreement updates

## 9. Liability & Disclaimers

### Service "As Is"

SafeSpeak is provided "as is" without warranties of any kind. We do not guarantee:

- Uninterrupted service availability
- Perfect accuracy in all scenarios
- Suitability for specific use cases

### Limitation of Liability

- Maximum liability limited to fees paid in previous 12 months
- No liability for indirect or consequential damages
- No liability for content moderation decisions

### Indemnification

Users agree to indemnify SafeSpeak against claims arising from:

- Misuse of the API
- Content moderation decisions
- Violation of applicable laws

## 10. Updates & Changes

### Policy Updates

- Changes announced 30 days in advance
- Major changes may require user consent
- Continued use implies acceptance

### API Changes

- Deprecation notices provided 90 days in advance
- Migration guides provided
- Backward compatibility maintained where possible

## 11. Contact Information

- **General Support**: support@safespeak.ai
- **Technical Issues**: tech@safespeak.ai
- **Privacy Concerns**: privacy@safespeak.ai
- **Business Inquiries**: business@safespeak.ai

## 12. Governing Law

This agreement is governed by applicable international data protection and consumer protection laws. Disputes shall be resolved through binding arbitration in accordance with UNCITRAL rules.

---

_Last updated: October 8, 2025_
_Version: 1.0_
