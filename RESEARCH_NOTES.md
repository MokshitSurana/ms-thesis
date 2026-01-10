# Research Notes: RQ1 - Effect of Note Structure

## Research Context

### Why This Matters

Clinical notes are highly structured documents with conventional sections (Chief Complaint, HPI, Physical Exam, etc.). This structure:
- Helps clinicians quickly locate information
- Reflects the clinical reasoning process
- Follows standardized templates (e.g., SOAP notes)

**Key Question:** Do LLMs leverage this structure, or do they work equally well with unstructured text?

### Implications

**If structure matters significantly:**
- Healthcare systems should maintain structured documentation standards
- LLM developers should design structure-aware models
- Data preprocessing should preserve formatting

**If structure doesn't matter:**
- Legacy unstructured notes can be processed effectively
- Simpler data pipelines (no structure preservation needed)
- Models are truly learning semantic understanding, not pattern matching

---

## Experimental Design Rationale

### Model Selection

#### General Models (BART-CNN, PEGASUS)
- **Trained on:** News articles, general domain text
- **Strength:** Strong general summarization capability
- **Weakness:** No medical knowledge, may rely heavily on structure

#### Medical Models (Clinical-T5, BioGPT)
- **Trained on:** Medical literature, clinical notes
- **Strength:** Medical vocabulary, clinical reasoning
- **Hypothesis:** More robust to structure loss due to domain knowledge

### Structure Variants Explained

#### 1. Original
Keeps authentic clinical note structure:
```
Chief Complaint:
Chest pain

History of Present Illness:
Patient is a 65 y/o male...
```
**Tests:** Baseline performance with full structure

#### 2. No Headers
Removes section labels but keeps organization:
```
Chest pain

Patient is a 65 y/o male...
```
**Tests:** Do models use headers as cues, or just content proximity?

#### 3. No Formatting
Continuous text, no structure markers:
```
Chest pain Patient is a 65 y/o male with acute onset...
```
**Tests:** Maximum structure degradation - pure semantic understanding

#### 4. Shuffled
Sections in random order:
```
Physical Exam: BP 140/90...
Chief Complaint: Chest pain
Brief Hospital Course: Patient underwent...
History: 65 y/o male...
```
**Tests:** Do models rely on conventional note organization?

---

## Hypotheses & Predictions

### H1: Structure Importance
**Hypothesis:** Performance decreases as structure is removed
```
Original > No Headers > No Formatting
```

**Rationale:**
- Headers provide semantic boundaries
- Formatting aids in information extraction
- Models trained on structured text may rely on these cues

**Metrics to watch:**
- ROUGE-L (captures structural similarity)
- Absolute ROUGE-1 drop

### H2: Medical Models More Robust
**Hypothesis:** Medical models show smaller performance drops
```
Clinical-T5_drop < BART-CNN_drop
```

**Rationale:**
- Medical models understand clinical semantics
- Less reliant on surface-level structure
- Trained on varied clinical documentation styles

**Metrics to watch:**
- Percentage drop from original to no_formatting
- Variance across structure types

### H3: Shuffling Disrupts Severely
**Hypothesis:** Shuffled sections hurt performance most
```
Shuffled << No Formatting
```

**Rationale:**
- Clinical reasoning follows logical flow
- Models may learn temporal/causal patterns
- Disrupting order breaks narrative coherence

**Metrics to watch:**
- Shuffled vs. No Formatting comparison
- ROUGE-2 (bigram overlap, sensitive to order)

---

## Expected Challenges

### 1. Model-Specific Issues
- **Clinical-T5:** May not be available on HuggingFace → Use alternatives
- **BioGPT:** Decoder-only, may need different generation strategy
- **Memory:** Large models may require batch_size=1

### 2. Evaluation Issues
- **BERTScore:** Very slow on 1000 samples → Consider disabling or sampling
- **Reference quality:** MIMIC summaries vary in quality
- **Length mismatch:** Generated vs. reference length differences

### 3. Data Issues
- **Structure variability:** Not all notes follow same template
- **Missing sections:** Some notes lack certain sections
- **Noise:** OCR errors, abbreviations, formatting artifacts

---

## Analysis Plan

### Primary Analysis

**1. Main Effect of Structure**
```python
# ANOVA or Friedman test
structure_effect = df.groupby('structure_variant')['rouge1'].mean()

# Post-hoc pairwise comparisons
from scipy.stats import friedmanchisquare
stat, p = friedmanchisquare(original, no_headers, no_formatting, shuffled)
```

**2. Model × Structure Interaction**
```python
# Two-way ANOVA
import statsmodels.api as sm
from statsmodels.formula.api import ols

model = ols('rouge1 ~ C(model) + C(structure_variant) + C(model):C(structure_variant)',
            data=df).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
```

**3. Robustness Analysis**
```python
# Calculate drop percentage for each model
for model in models:
    original_score = df[(df['model']==model) & (df['structure_variant']=='original')]['rouge1'].values[0]
    worst_score = df[df['model']==model]['rouge1'].min()
    robustness = (original_score - worst_score) / original_score * 100
    print(f"{model}: {robustness:.1f}% drop")
```

### Secondary Analysis

**1. Error Analysis**
- Manually review 20 samples with largest performance drops
- Identify what information is lost without structure
- Categorize errors (missing diagnosis, wrong treatment, etc.)

**2. Length Analysis**
```python
# Do longer notes benefit more from structure?
df['note_length'] = df['text'].str.split().str.len()
correlation = df.groupby('structure_variant').apply(
    lambda x: x[['note_length', 'rouge1']].corr().iloc[0,1]
)
```

**3. Section Importance**
- Which sections are most critical? (Chief Complaint vs. Social History)
- Create variants with selective section removal

---

## Writing the Results Section

### Table 1: Overall Performance
| Model | Original | No Headers | No Formatting | Shuffled | Avg |
|-------|----------|------------|---------------|----------|-----|
| BART-CNN | 0.42 | 0.38 | 0.33 | 0.29 | 0.36 |
| Clinical-T5 | 0.45 | 0.42 | 0.39 | 0.36 | 0.41 |

### Table 2: Performance Drop
| Model | Drop (Original → No Formatting) |
|-------|----------------------------------|
| BART-CNN | 21.4% |
| Clinical-T5 | 13.3% |

### Figure 1: Structure Degradation Effect
Line plot showing ROUGE-1 across structure variants for each model

### Figure 2: Heatmap
Model × Structure heatmap showing ROUGE scores

### Key Findings (Expected)

1. **Structure matters significantly** (p < 0.001)
   - Average 18% drop from original to no_formatting
   - Headers alone account for ~40% of structure benefit

2. **Medical models more robust**
   - Clinical-T5: 13% drop vs. BART-CNN: 21% drop
   - Suggests domain knowledge compensates for structure loss

3. **Shuffling severely disrupts**
   - Shuffled performs worse than no_formatting
   - Indicates models leverage conventional note organization

---

## Discussion Points

### Clinical Implications
- **For EHR systems:** Maintain structured templates
- **For model deployment:** Preprocessing should preserve structure
- **For legacy data:** May need structure reconstruction

### Technical Insights
- Medical domain adaptation provides robustness
- Structure serves as inductive bias
- Future models should explicitly model clinical note structure

### Limitations
- Single dataset (MIMIC-IV)
- Limited to discharge summaries
- Evaluation on automatic metrics only

### Future Work
- Test on other clinical tasks (NER, relation extraction)
- Investigate section-specific importance
- Design structure-aware models

---

## Thesis Outline

### Chapter: RQ1 - Effect of Note Structure

#### 3.1 Introduction (2 pages)
- Clinical notes are structured documents
- Research question and hypotheses
- Significance

#### 3.2 Methods (3 pages)
- Dataset description
- Models (general vs. medical)
- Structure variants
- Evaluation metrics

#### 3.3 Results (4 pages)
- Main effect of structure (Table 1, Figure 1)
- Model comparison (Table 2, Figure 2)
- Statistical tests
- Error analysis

#### 3.4 Discussion (3 pages)
- Interpretation of findings
- Comparison with related work
- Clinical implications
- Limitations

#### 3.5 Conclusion (1 page)
- Summary of key findings
- Answer to research question
- Transition to next RQ

---

## References to Review

### Structure in Clinical NLP
- Van Vleck & Elhadad (2010) - Corpus analysis of clinical notes
- Haug et al. (1995) - Clinical note structure standards
- Cohen et al. (2014) - Structural variations in EHRs

### Summarization of Clinical Notes
- Pivovarov & Elhadad (2015) - Automated summarization
- Zhang et al. (2018) - Clinical BERT applications
- Adams et al. (2021) - GPT-3 for medical summarization

### Structure-Aware Models
- Lee et al. (2020) - Hierarchical models for documents
- Guo et al. (2021) - Section-aware summarization
- Krishna et al. (2021) - Long document understanding

---

## Quick Wins for Paper

### Supplementary Material
- Include example transformations for each variant
- Show sample outputs from each model
- Provide error analysis spreadsheet

### Visualizations
- Create attention heatmaps (if possible)
- Show section importance via ablation
- Length vs. performance scatter plots

### Validation
- Inter-annotator agreement on manual error analysis
- Bootstrap confidence intervals for metrics
- Cross-validation if data permits

Good luck with your thesis! 📊
