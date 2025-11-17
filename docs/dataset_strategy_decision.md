# SafeSpeak Dataset Strategy Decision

## Decision: Drop English Datasets

**Date:** October 6, 2025
**Decision Maker:** SafeSpeak Development Team

### Context

After comprehensive analysis of 288,071 samples across 13 datasets, we identified significant challenges with English datasets that impact the multilingual toxicity detection goals of SafeSpeak.

### English Dataset Analysis

- **hatexplain_en**: 20,148 samples (balanced, but English-only)
- **jigsaw_en**: 159,571 samples (severely imbalanced: 89.8% Neutral vs 10.2% Toxic)

### Issues Identified

1. **Severe Class Imbalance**: Jigsaw dataset has 10:1 ratio (Neutral:Toxic)
2. **Language Focus Mismatch**: SafeSpeak targets FR/AR/Darija languages
3. **Transfer Learning Concerns**: English-trained models may not generalize well to Arabic/French dialects
4. **Resource Allocation**: Training on English data diverts resources from target languages

### Decision Rationale

- **Focus on Target Languages**: Prioritize Arabic, French, and Darija datasets
- **Quality over Quantity**: Better to have balanced, relevant data than massive imbalanced English data
- **Multilingual Expertise**: Build models specifically tuned for target languages rather than relying on English transfer learning
- **Resource Efficiency**: Concentrate compute resources on relevant datasets

### Impact

- **Samples Reduced**: From 288,071 to ~108,351 (62% reduction)
- **Languages Retained**: Arabic (84,478), French (23,872), Darija (1)
- **Label Balance**: Improved overall balance without English dominance
- **Training Focus**: Direct attention to multilingual Arabic/French toxicity patterns

### Remaining Datasets

**Arabic Datasets:**

- algd_toxicity_dz (14,150 samples)
- arabizi_offensive_lang (7,335 samples)
- armi_ar_train (7,866 samples)
- base_donnee_hate_speech_ar (38,544 samples)
- hatecheck_arabic (3,570 samples)
- narabizi_treebank (1,287 samples)
- toxic_arabic_tweets (5,758 samples)
- t_hsab_tunisian (5,968 samples)

**French Datasets:**

- hatecheck_french (3,718 samples)
- hatexplain_fr (20,148 samples)

**Multilingual:**

- sample_toxicity (8 samples - mixed languages)

### Next Steps

Proceed with focused multilingual dataset development and model training for FR/AR/Darija toxicity detection.</content>
<parameter name="filePath">c:\Users\GIGABYTE\projects\SafeSpeak - NLP\docs\dataset_strategy_decision.md
