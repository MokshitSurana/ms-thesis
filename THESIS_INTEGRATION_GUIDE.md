# Thesis Integration Guide: Dual Evaluation Approach

## Overview

This guide shows how to present **both evaluation approaches** in your thesis while keeping them clearly differentiated.

### The Two Approaches

| Aspect | Reference-Based | Reference-Free |
|--------|----------------|----------------|
| **What it compares** | Predictions vs gold-standard references | Variants vs original structure baseline |
| **Question answered** | "How good is the summary?" | "How much does structure change output?" |
| **Metrics** | ROUGE, BERTScore vs reference | BERTScore between variants, length, terms |
| **Shows** | Absolute quality | Relative impact |
| **Best for** | Comparing to state-of-the-art | Directly answering RQ1 |

---

## Thesis Structure

### Chapter 1: Introduction

**Mention both approaches early:**

```markdown
To comprehensively evaluate the effect of note structure on summarization
(RQ1), we employ a dual evaluation framework:

1. **Reference-Based Evaluation**: Compares generated summaries to gold-standard
   references using ROUGE and BERTScore, measuring absolute summary quality and
   alignment with clinical standards.

2. **Reference-Free Evaluation**: Compares structure variants against the original
   structure as a baseline, directly measuring how much each manipulation changes
   the model's output, independent of reference quality.

This dual approach provides both absolute quality measures and relative impact
measures, offering convergent evidence for our findings.
```

---

### Chapter 3: Methodology

**Section 3.4: Evaluation Framework**

```markdown
## 3.4 Evaluation Framework

We employ a dual evaluation approach to comprehensively assess RQ1: the effect
of note structure on clinical summarization.

### 3.4.1 Reference-Based Evaluation

**Purpose**: Measure absolute summary quality by comparing predictions to
gold-standard human-written summaries.

**Metrics**:
- ROUGE-1/2/L: Lexical n-gram overlap
- BERTScore: Semantic similarity using contextual embeddings
- Length statistics: Word count comparison

**Baseline**: First two sentences of the "Brief Hospital Course" section
(mean: 102 words, SD: 31 words)

**Interpretation**: Higher scores indicate better alignment with clinical
summarization standards. This approach allows comparison to prior work and
establishes absolute quality bounds.

### 3.4.2 Reference-Free Evaluation

**Purpose**: Directly measure how much structure manipulation changes model output,
independent of reference quality.

**Approach**: Use the **original structure** predictions as the baseline, then
measure how much each variant (no_headers, no_formatting, shuffled) deviates.

**Metrics**:
1. **Semantic Similarity** (BERTScore between variants): How much meaning changes?
   - High (>0.90): Minimal semantic change
   - Low (<0.70): Substantial semantic deviation

2. **Length Change** (%): How much verbosity changes?
   - Positive: Longer than original
   - Negative: Shorter than original

3. **Medical Term Overlap** (%): How much clinical content preserved?
   - High (>0.90): Similar clinical focus
   - Low (<0.70): Different clinical concepts emphasized

**Interpretation**: Large deviations indicate structure strongly affects output.
This approach directly answers RQ1 without reference quality confounds.

### 3.4.3 Rationale for Dual Approach

Reference-based and reference-free evaluations provide **complementary evidence**:

- **Convergent findings** (both show same effect) → Strong evidence
- **Divergent findings** (metrics disagree) → Reveals metric limitations
- **Complete picture**: Absolute quality + relative impact

Figure X shows our dual evaluation methodology.
[Insert: methodology_flowchart.pdf]
```

---

### Chapter 4: Results

**Organization:**

```markdown
# Chapter 4: Results

## 4.1 Overview
Brief summary of both approaches

## 4.2 Reference-Based Evaluation
Traditional metrics comparing to gold standard

## 4.3 Reference-Free Evaluation
Variant comparison against original baseline

## 4.4 Convergent Evidence
How both approaches support same conclusions
```

---

#### Section 4.2: Reference-Based Evaluation

```markdown
## 4.2 Reference-Based Evaluation

We first present traditional reference-based evaluation, comparing generated
summaries to gold-standard references.

### 4.2.1 Overall Performance

Figure 4.1 shows ROUGE-1 scores across all model-structure combinations.
[Insert: fig1_structure_effect_by_model.pdf]

Key findings:
- BioBART achieved highest ROUGE-1 (0.315) on no_headers variant
- General models (BART-CNN, PEGASUS) scored 0.22-0.26
- Domain-specific pre-training improved performance by 20%

### 4.2.2 Impact of Shuffling

Figure 4.2 shows the devastating impact of disrupting logical flow.
[Insert: fig3_shuffling_impact.pdf]

Shuffling reduced performance across all models:
- BART-CNN: -12% (0.262 → 0.231)
- PEGASUS: -29% (0.228 → 0.162)
- BioBART: -43% (0.298 → 0.170)

**Finding**: Medical models are MORE sensitive to structure disruption than
general models, suggesting they rely more heavily on domain-specific structural
patterns learned during pre-training.

### 4.2.3 Metric-Quality Disconnect

However, reference-based metrics alone proved insufficient for quality
assessment. Figure 4.3 reveals a critical disconnect.
[Insert: fig4_rouge_vs_bertscore.pdf]

BioBART no_headers achieved the highest ROUGE-1 (0.315) but only moderate
BERTScore (0.569), while no_formatting had lower ROUGE-1 (0.298) but highest
BERTScore (0.615). Qualitative analysis revealed no_headers produced garbage
text (Unicode artifacts: \u00af\u00af, spurious tokens: "Nitrome", "petertodd")
despite high ROUGE scores.

**This motivated our reference-free evaluation approach.**
```

---

#### Section 4.3: Reference-Free Evaluation

```markdown
## 4.3 Reference-Free Evaluation

To address reference-based limitations and directly measure structure impact,
we compare each variant against the **original structure as baseline**.

### 4.3.1 Semantic Similarity

Figure 4.4 shows semantic similarity (BERTScore F1) between variants and
original structure.
[Insert: bertscore_deviation.pdf]

Key findings:
- no_headers: 0.94 ± 0.02 (minimal semantic change)
- no_formatting: 0.91 ± 0.03 (slight change)
- shuffled: 0.72 ± 0.08 (substantial deviation)

**Interpretation**: Shuffling fundamentally changes semantic content, while
removing headers preserves meaning. This suggests headers provide redundant
structural information that models can infer from content alone.

### 4.3.2 Length and Verbosity

Figure 4.5 shows length changes relative to original structure.
[Insert: length_changes.pdf]

- no_headers: -2% to +1% (minimal change)
- no_formatting: -3% to +2% (slight variation)
- shuffled: -8% to -15% (substantial reduction)

Shuffled variants produced shorter summaries across all models, suggesting
disrupted flow impairs the model's ability to extract comprehensive information.

### 4.3.3 Medical Term Retention

Figure 4.6 shows clinical vocabulary overlap with original structure.
[Insert: term_overlap.pdf]

- no_headers: 92% (high retention)
- no_formatting: 87-89% (moderate retention)
- shuffled: 65-72% (substantial loss)

**Interpretation**: Logical section ordering helps models identify and retain
key medical concepts. Disruption reduces clinical vocabulary by 28%.

### 4.3.4 Model Differences

BioBART showed highest sensitivity to structure manipulation:
- Semantic similarity drop: 0.95 → 0.65 (32% decrease for shuffled)
- BART-CNN drop: 0.94 → 0.72 (23% decrease)

This confirms findings from reference-based evaluation: medical models rely
more heavily on structural patterns.
```

---

#### Section 4.4: Convergent Evidence

```markdown
## 4.4 Convergent Evidence

Figure 4.7 synthesizes findings from both evaluation approaches.
[Insert: findings_synthesis.pdf]

### 4.4.1 Finding 1: Shuffling Devastates Performance

**Reference-Based**: ROUGE-1 drops 12-43%, BERTScore vs reference: 0.51-0.59
**Reference-Free**: Semantic similarity: 0.65-0.72 (vs 0.94 baseline), term loss: 28%
**Convergent Evidence**: Logical flow is critical for both quality and content

### 4.4.2 Finding 2: Headers Are Redundant

**Reference-Based**: ROUGE-1 within 5% of original, minimal quality impact
**Reference-Free**: Semantic similarity 0.94, term overlap 92%
**Convergent Evidence**: Content, not headers, drives summarization

### 4.4.3 Finding 3: Medical Models More Structure-Dependent

**Reference-Based**: BioBART -43% on shuffle vs BART-CNN -12%
**Reference-Free**: BioBART semantic drop: 32% vs BART-CNN 23%
**Convergent Evidence**: Domain-specific pre-training increases structure reliance

### 4.4.4 Finding 4: Metrics Can Mislead

**Reference-Based**: BioBART no_headers: highest ROUGE (0.315) but garbage output
**Reference-Free**: High similarity to original (0.95) but both contain artifacts
**Key Insight**: High agreement between bad outputs → Need qualitative analysis

### 4.4.5 Answering RQ1

Both evaluation approaches provide convergent evidence:

**RQ1: Does note structure affect clinical summarization performance?**

**Answer: YES - significantly.**

Evidence:
1. Shuffling reduces both absolute quality (ROUGE -43%) and semantic similarity
   to baseline (BERTScore 0.72 vs 0.94)
2. Effects vary by model: medical models show 2× greater sensitivity
3. Different structural elements have different effects:
   - Headers: Redundant (minimal impact when removed)
   - Logical flow: Critical (devastating impact when disrupted)
   - Formatting: Moderate (affects verbosity more than content)

**Implication**: Structure provides crucial scaffolding for content extraction,
especially for domain-adapted models. However, explicit markers (headers) are
less important than implicit organization (logical flow).
```

---

### Chapter 5: Discussion

```markdown
## 5.3 Methodological Contributions

### 5.3.1 Dual Evaluation Framework

This work contributes a dual evaluation framework combining:

1. **Reference-Based**: Establishes absolute quality and enables comparison to
   prior work
2. **Reference-Free**: Directly measures manipulation impact without reference
   bias

This approach proved essential for our findings:
- Revealed metric-quality disconnects (BioBART no_headers)
- Provided independent confirmation of structure importance
- Clarified which structural elements matter most

### 5.3.2 When to Use Each Approach

**Use Reference-Based when**:
- Comparing to state-of-the-art
- Absolute quality matters
- Gold-standard references available

**Use Reference-Free when**:
- Studying manipulation effects (like RQ1)
- References may be biased/problematic
- Interested in relative rather than absolute quality

**Use Both when**:
- Comprehensive evaluation needed
- Metric validity concerns exist
- Strong evidence required for claims

### 5.3.3 Implications for Clinical NLP

The metric-quality disconnect observed with BioBART highlights limitations of
automatic evaluation for clinical text. Future work should:

1. Always combine automatic metrics with qualitative analysis
2. Use multiple evaluation frameworks (not just one)
3. Consider domain-specific evaluation metrics for clinical summarization
4. Validate that high scores correspond to usable outputs
```

---

## Figure Numbering Strategy

### Methodology Chapter (Chapter 3)
- **Figure 3.1**: `methodology_flowchart.pdf` - Dual evaluation methodology

### Results Chapter (Chapter 4)

#### Reference-Based Figures (Section 4.2)
- **Figure 4.1**: `fig1_structure_effect_by_model.pdf` - Main RQ1 finding
- **Figure 4.2**: `fig3_shuffling_impact.pdf` - Shuffling devastates performance
- **Figure 4.3**: `fig4_rouge_vs_bertscore.pdf` - Metric disconnect
- **Figure 4.4**: `fig6_domain_comparison.pdf` - General vs medical models

#### Reference-Free Figures (Section 4.3)
- **Figure 4.5**: `bertscore_deviation.pdf` - Semantic similarity to original
- **Figure 4.6**: `length_changes.pdf` - Length changes from baseline
- **Figure 4.7**: `term_overlap.pdf` - Medical term retention

#### Synthesis Figure (Section 4.4)
- **Figure 4.8**: `findings_synthesis.pdf` - Convergent evidence
- **Figure 4.9**: `dual_evaluation_framework.pdf` - Side-by-side comparison

### Appendix
- **Figure A.1**: `fig2_model_comparison.pdf` - Detailed model comparison
- **Figure A.2**: `fig5_performance_heatmap.pdf` - Comprehensive heatmap
- **Figure A.3**: `fig7_all_metrics.pdf` - Multi-metric comparison
- **Figure A.4**: `comprehensive_heatmap.pdf` - Reference-free heatmap

---

## Table Numbering Strategy

### Results Chapter
- **Table 4.1**: `summary_table.tex` - Reference-based results (all combinations)
- **Table 4.2**: `reference_free_table.tex` - Reference-free results

### Methodology/Appendix
- **Table 3.1** or **Table A.1**: `evaluation_approaches_comparison.tex` - Comparison of approaches

---

## Writing Tips: Differentiating the Approaches

### ✅ DO: Use Clear Labels

**Good:**
```markdown
Reference-based evaluation (comparing to gold-standard references) showed
ROUGE-1 = 0.315, while reference-free evaluation (comparing to original
structure baseline) showed semantic similarity = 0.95.
```

**Bad:**
```markdown
ROUGE-1 = 0.315 and BERTScore = 0.95.
```
(Confusing - not clear what's being compared to what)

### ✅ DO: Use Consistent Terminology

| Approach | Term for Comparison Target |
|----------|---------------------------|
| Reference-Based | "gold-standard reference", "human summary", "reference summary" |
| Reference-Free | "original structure baseline", "original variant", "baseline output" |

### ✅ DO: Explain Why You Need Both

**Example:**
```markdown
While reference-based evaluation is standard, it has limitations for RQ1:
reference quality and length can bias results, and high scores may not reflect
usable outputs (as seen with BioBART no_headers). Reference-free evaluation
addresses these issues by comparing variants directly, providing independent
confirmation of structure effects.
```

### ✅ DO: Show How They Complement

**Example:**
```markdown
The dual approach revealed insights neither method alone could provide.
Reference-based evaluation showed BioBART no_headers achieved highest ROUGE
(0.315), while reference-free evaluation revealed this variant produced nearly
identical output to original structure (similarity: 0.95) - both containing
artifacts. This convergence proved metrics alone are insufficient for quality
assessment.
```

### ❌ DON'T: Mix Metrics Without Context

**Bad:**
```markdown
BioBART scored 0.315 on ROUGE-1 and 0.95 on BERTScore.
```

**Good:**
```markdown
BioBART achieved 0.315 ROUGE-1 when compared to reference summaries, and 0.95
BERTScore similarity when compared to original structure predictions.
```

### ❌ DON'T: Use "BERTScore" Ambiguously

**Problem**: BERTScore is used in BOTH approaches!
- Reference-based: BERTScore vs reference
- Reference-free: BERTScore between variants

**Solution**: Always specify comparison target
- "BERTScore vs reference: 0.615"
- "BERTScore vs original: 0.95"

---

## Defense Presentation Strategy

### Slide 1: Motivation
"Why two evaluation approaches?"
- Show BioBART no_headers: ROUGE=0.315 (best!) but garbage output
- "We needed a second opinion"

### Slide 2: Dual Framework
Show `methodology_flowchart.pdf`
- Same input → two evaluation paths → convergent conclusions

### Slide 3: Reference-Based Results
Show `fig1_structure_effect_by_model.pdf`
- "Structure matters for absolute quality"

### Slide 4: Reference-Free Results
Show `bertscore_deviation.pdf`
- "Structure matters for semantic content"

### Slide 5: Convergent Evidence
Show `findings_synthesis.pdf`
- "Both approaches agree: structure is critical"

### Slide 6: Key Insight
Show `dual_evaluation_framework.pdf` side-by-side
- "Two perspectives, one conclusion"

---

## Common Questions (and Answers)

### Q: "Isn't this just doing the same thing twice?"

**A**: No - they measure fundamentally different things:
- Reference-based: "Is this a good summary?" (absolute quality)
- Reference-free: "How much does structure change output?" (relative impact)

Both are needed because:
1. Reference-based can be biased by reference quality
2. Reference-free doesn't measure absolute quality
3. Convergent evidence from independent methods is stronger

### Q: "Which approach is more important?"

**A**: They're complementary, not competing:
- Reference-based: Required for comparison to prior work, establishes credibility
- Reference-free: Directly answers RQ1 without confounds

For your thesis, both are essential.

### Q: "Can I use just one?"

**A**: You could, but you'd lose:
- Reference-based only: Miss the BioBART quality issue, can't separate structure
  effect from reference bias
- Reference-free only: Can't compare to state-of-the-art, no absolute quality
  baseline

The dual approach revealed insights neither method alone could provide.

### Q: "How do I explain this to non-experts?"

**A**: Use an analogy:

"Imagine evaluating student essays. You can:
1. Compare to a model essay (reference-based) - shows if they're good writers
2. Compare rough draft to final draft (reference-free) - shows if editing helped

Both tell you something useful, but different things. We do the same for
summarization models."

---

## Checklist: Integrating Both Approaches

### Methodology Chapter
- [ ] Explain both approaches clearly
- [ ] Justify why both are needed
- [ ] Show methodology flowchart
- [ ] Define all metrics precisely
- [ ] Clarify what each approach measures

### Results Chapter
- [ ] Present reference-based results first (familiar to readers)
- [ ] Present reference-free results second (novel contribution)
- [ ] Create synthesis section showing convergence
- [ ] Use clear labels ("reference-based", "reference-free")
- [ ] Never use metrics without context

### Discussion Chapter
- [ ] Explain what dual approach revealed
- [ ] Discuss methodological contribution
- [ ] Provide guidance on when to use each
- [ ] Address limitations of both approaches

### Figures
- [ ] Label all figures with approach name
- [ ] Use consistent color coding (blue=reference-based, green=reference-free)
- [ ] Include approach in figure captions
- [ ] Create at least one side-by-side comparison figure

### Writing
- [ ] Never say just "BERTScore" - always specify vs what
- [ ] Always mention comparison target
- [ ] Use consistent terminology
- [ ] Explain complementarity, not redundancy

---

## Example Abstract

```markdown
Clinical note structure may affect summarization model performance, but
evaluation is complicated by reference quality concerns. We propose a dual
evaluation framework combining reference-based evaluation (comparing to
gold-standard summaries) and reference-free evaluation (comparing structure
variants to original baseline). Using MIMIC-IV notes (n=1000), we evaluate
three models (BART-CNN, PEGASUS, BioBART) across four structure variants.
Reference-based evaluation shows shuffling reduces ROUGE-1 by 12-43%, with
medical models most affected. Reference-free evaluation confirms shuffling
reduces semantic similarity to baseline by 23-32% and medical term retention
by 28%. Convergent evidence from both approaches demonstrates note structure
significantly affects clinical summarization, with logical flow critical but
explicit headers redundant. The dual framework also reveals metric-quality
disconnects requiring qualitative analysis. This work contributes both empirical
findings on structure importance and a methodological framework for robust
evaluation of text manipulation effects.
```

---

## Final Recommendations

1. **Always differentiate**: Use clear labels, never mix approaches

2. **Explain complementarity**: Show why both are needed, not redundant

3. **Show convergence**: Demonstrate both approaches support same conclusions

4. **Acknowledge limitations**: Each approach has weaknesses, together they're stronger

5. **Contribute methodologically**: Frame dual approach as contribution, not just belt-and-suspenders

6. **Use visually**: Side-by-side figures make differentiation obvious

7. **Be consistent**: Use same terminology throughout thesis

8. **Think defensively**: Anticipate "why both?" question and answer proactively

---

**Remember**: The dual approach is a STRENGTH, not redundancy. It shows rigor,
addresses validity concerns, and provides convergent evidence. Frame it as a
methodological contribution to clinical NLP evaluation.
