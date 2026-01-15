# Reference-Free Analysis Guide

## Overview

This guide explains how to perform reference-free analysis that compares structure variants against the **original structure as a baseline**, rather than comparing against gold-standard references.

### Why Reference-Free Analysis?

**Traditional (Reference-Based) Evaluation:**
- Compares predictions to gold-standard human summaries
- Measures: "How close is the model to the ideal?"
- Metric: ROUGE, BERTScore vs reference

**Reference-Free (Variant Comparison) Evaluation:**
- Compares structure variants to the original structure
- Measures: "How much does structure manipulation change the output?"
- Metrics: Semantic similarity, length changes, content preservation

### Key Benefits for Your Thesis

1. **Direct RQ1 Answer**: Shows exactly how much each structure manipulation affects output
2. **No Reference Bias**: Avoids issues with reference quality or length mismatch
3. **Complementary Evidence**: Strengthens findings when combined with traditional metrics
4. **Qualitative Insights**: Reveals what content changes when structure changes

---

## Step 1: Generate Prediction Files

First, you need to save all model predictions for each structure variant.

### Option A: Run Modified Experiment Script

```bash
python experiment_save_predictions.py
```

**What this does:**
- Loads each model (BART-CNN, PEGASUS, BioBART)
- Generates predictions for all 4 structure variants (original, no_headers, no_formatting, shuffled)
- Saves predictions to `predictions/` directory
- Creates 12 JSON files (3 models × 4 variants)

**Expected output:**
```
predictions/
├── bart-cnn_original_predictions.json
├── bart-cnn_no_headers_predictions.json
├── bart-cnn_no_formatting_predictions.json
├── bart-cnn_shuffled_predictions.json
├── pegasus_original_predictions.json
├── pegasus_no_headers_predictions.json
├── pegasus_no_formatting_predictions.json
├── pegasus_shuffled_predictions.json
├── biobart_original_predictions.json
├── biobart_no_headers_predictions.json
├── biobart_no_formatting_predictions.json
└── biobart_shuffled_predictions.json
```

**⏱️ Time estimate:** 2-4 hours (depending on GPU)

### Option B: Modify Existing Experiment Output

If you already ran experiments, you can extract predictions from existing output:

```python
# Add this to your experiment.py after generating predictions
import json
from pathlib import Path

predictions_dir = Path('predictions')
predictions_dir.mkdir(exist_ok=True)

# Save predictions for this model-variant combination
output_file = predictions_dir / f'{model_name}_{structure_variant}_predictions.json'
with open(output_file, 'w') as f:
    json.dump(predictions_with_metadata, f, indent=2)
```

---

## Step 2: Run Reference-Free Analysis

Once prediction files exist:

```bash
python reference_free_analysis.py
```

**What this analyzes:**

### 1. Semantic Similarity (BERTScore between variants)
- Compares each variant's predictions to original structure predictions
- **High score (>0.9)**: Variant produces very similar summaries
- **Low score (<0.7)**: Significant semantic deviation

### 2. Length Changes
- Calculates percent change in summary length
- Positive = longer than original, Negative = shorter
- Shows verbosity differences

### 3. Medical Term Overlap
- Extracts key medical terms from predictions
- Measures retention of clinical vocabulary
- Shows content preservation

---

## Step 3: Interpret Results

### Generated Visualizations

**1. `bertscore_deviation.png`**
- Shows semantic similarity between variants and original
- **Thesis interpretation**: "Structure manipulation changed semantic content by X%"
- **Key insight**: Shuffled has lowest similarity → structure matters for meaning

**2. `length_changes.png`**
- Shows how length changes across variants
- **Thesis interpretation**: "Removing headers increased verbosity by X%"
- **Key insight**: Formatting constraints affect summary length

**3. `term_overlap.png`**
- Shows medical term retention
- **Thesis interpretation**: "Shuffling reduced clinical term usage by X%"
- **Key insight**: Structure helps models focus on key medical concepts

**4. `comprehensive_heatmap.png`**
- All metrics at once
- Easy pattern recognition
- Shows which models are most/least affected by structure

### Output Files

**1. `reference_free_comparison.csv`**
- Numerical results for all combinations
- Import into thesis for tables

**2. `reference_free_table.tex`**
- LaTeX-formatted table
- Ready to copy-paste into thesis

---

## Step 4: Integrate into Thesis

### Recommended Structure

#### Chapter 4: Results

**Section 4.1: Traditional Evaluation (Reference-Based)**
- Use: ROUGE scores, BERTScore vs gold standard
- Figures: `fig1_structure_effect_by_model.png`, `fig3_shuffling_impact.png`

**Section 4.2: Comparative Analysis (Reference-Free)**
- Use: Variant comparison against original baseline
- Figures: `bertscore_deviation.png`, `length_changes.png`, `term_overlap.png`
- Key message: "Structure manipulation changes output by X% even when ROUGE scores are similar"

**Section 4.3: Model-Specific Analysis**
- BioBART quality issues
- Figure: `fig4_rouge_vs_bertscore.png`

### Sample Thesis Text

```markdown
To complement traditional reference-based evaluation, we performed a
reference-free analysis comparing structure variants against the original
structure as a baseline. This approach directly measures how much each
manipulation changes the model's output, independent of reference quality.

Figure X shows semantic similarity (BERTScore F1) between variant outputs
and original structure outputs. The shuffled variant showed the greatest
deviation (BERTScore F1 = 0.72 ± 0.08), indicating that disrupting logical
flow fundamentally changes the semantic content of generated summaries.
In contrast, no_headers variant remained highly similar to original
(BERTScore F1 = 0.94 ± 0.03), suggesting headers provide redundant
information for content selection.

Medical term analysis (Figure Y) revealed that shuffled sections reduced
clinical vocabulary retention by 28%, while formatting changes had minimal
impact (<5% change). This suggests that logical section ordering is crucial
for models to identify and retain key medical concepts.
```

---

## Troubleshooting

### Issue: "bert_score not installed"
```bash
pip install bert-score
```

The script will run without BERTScore but skip semantic similarity analysis. Length and term overlap will still work.

### Issue: "No prediction files found"
- Make sure you ran `experiment_save_predictions.py` first
- Check that `predictions/` directory exists and contains JSON files
- Verify file naming: `{model}_{variant}_predictions.json`

### Issue: Out of Memory
Edit `experiment_save_predictions.py`:
```python
BATCH_SIZE = 2  # Reduce from 4
max_samples=100  # Test with 100 samples first
```

---

## Technical Details

### BERTScore Model
- Uses: `microsoft/deberta-xlarge-mnli`
- Sampling: First 100 examples (representative, faster)
- Metrics: Precision, Recall, F1

### Medical Terms Detected
- Diagnosis, treatment, procedure
- Patient, symptoms, conditions
- Medications, admission/discharge
- History, examination
- Uses regex patterns (see `MEDICAL_TERMS` in script)

### Statistical Approach
- Compares distributions, not individual examples
- Mean values with confidence intervals
- Percent change from baseline
- N=1000 for statistical power (N=100 minimum)

---

## Expected Results

### Hypotheses

**H1: Shuffled variant will show lowest similarity**
- Rationale: Disrupts logical flow → different content selection
- Expected: BERTScore F1 < 0.80

**H2: Formatting changes will affect length but not content**
- Rationale: Same information, different verbosity
- Expected: High semantic similarity (>0.90), but length change >10%

**H3: Medical models more affected by structure**
- Rationale: Trained on structured clinical notes
- Expected: BioBART shows larger deviations than BART-CNN/PEGASUS

### Validation

Compare reference-free results to traditional ROUGE scores:
- If ROUGE and semantic similarity both change → **real impact**
- If ROUGE changes but semantic similarity doesn't → **metric artifact**
- If semantic similarity changes but ROUGE doesn't → **qualitative difference**

---

## Next Steps

After running the analysis:

1. ✓ Verify results align with traditional metrics
2. ✓ Identify any discrepancies (e.g., BioBART no_headers high ROUGE, low similarity)
3. ✓ Perform qualitative analysis on examples with large deviations
4. ✓ Integrate findings into thesis narrative
5. ✓ Use visualizations in defense presentation

---

## Citation

If you use this methodology, consider citing relevant papers:

- Zhang et al. (2020) - BERTScore: Evaluating Text Generation with BERT
- Reference-free evaluation approaches in summarization literature
- Clinical NLP evaluation methodologies

---

## Questions?

Common questions:

**Q: Should I use reference-based or reference-free evaluation?**
A: **Both!** They complement each other. Reference-based shows absolute quality, reference-free shows relative impact.

**Q: Which metric is more important for RQ1?**
A: Reference-free directly answers "does structure matter?" But reference-based provides credibility and comparison to prior work.

**Q: Can I use this for RQ2/RQ3?**
A: Yes! The same approach works for any manipulation (length, domain, etc.) - just use the baseline as reference.
