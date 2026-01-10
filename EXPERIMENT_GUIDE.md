# Experiment Execution Guide

## Step-by-Step Guide to Run Your Thesis Experiments

### Phase 1: Setup & Verification (15 minutes)

#### 1.1 Install Dependencies
```bash
pip install -r requirements.txt
```

#### 1.2 Download NLTK Data
```bash
python -c "import nltk; nltk.download('punkt')"
```

#### 1.3 Verify Setup
```bash
python test_setup.py
```

Expected output: All tests should pass ✓

#### 1.4 Run Quick Demo
```bash
python test_setup.py --demo
```

This demonstrates the full pipeline on a single example.

---

### Phase 2: Quick Testing (30 minutes)

#### 2.1 Test Structure Manipulation
```bash
python structure_manipulation.py
```

This shows how notes are transformed into different variants.

#### 2.2 Test with 3 Samples
```bash
python quick_start.py
```

This runs BART-CNN on 3 notes (similar to your original code).

#### 2.3 Small Experiment (10 samples)
```bash
python experiment.py
```

Tests BART-CNN on 10 samples with 2 structure variants.
**Time estimate:** ~5-10 minutes

---

### Phase 3: Pilot Study (2-3 hours)

Run a pilot with 100 samples to check everything works:

```python
from experiment import ExperimentRunner

runner = ExperimentRunner(n_samples=100)

# Test 2 models × 4 structures = 8 configurations
runner.run_full_experiment(
    models_to_test=['bart-cnn', 'clinical-t5'],
    variants_to_test=['original', 'no_headers', 'no_formatting', 'shuffled']
)

runner.create_comparison_table()
```

**Time estimate:** 2-3 hours (depends on GPU)

Review results in `results/summary_results.csv` before proceeding.

---

### Phase 4: Full Experiment (8-12 hours)

⚠️ **Run this overnight or on a compute cluster**

```bash
python experiment.py --full
```

This runs:
- All models (BART-CNN, PEGASUS, Clinical-T5, BioGPT)
- All structure variants (original, no_headers, no_formatting, shuffled)
- All 1000 samples
- **Total:** 4 models × 4 variants × 1000 samples = 16,000 inferences

**Time estimate:** 8-12 hours on GPU, 24-48 hours on CPU

#### Recommended: Run in Screen/Tmux

```bash
# Start screen session
screen -S thesis_experiment

# Run experiment
python experiment.py --full

# Detach: Ctrl+A, then D
# Reattach later: screen -r thesis_experiment
```

#### Monitor Progress

Results are saved incrementally in `results/detailed_results.json`. You can check progress:

```bash
# In another terminal
tail -f results/detailed_results.json
```

---

### Phase 5: Analysis (1-2 hours)

#### 5.1 Generate Visualizations

```bash
python visualize.py
```

This creates:
- `results/figures/all_rouge_metrics.png` - Comprehensive comparison
- `results/figures/rouge1_heatmap.png` - Performance heatmap
- `results/figures/structure_degradation_effect.png` - Impact analysis

#### 5.2 Statistical Summary

```python
from visualize import ResultsVisualizer

viz = ResultsVisualizer()
viz.generate_summary_stats()
```

#### 5.3 Custom Analysis

```python
import pandas as pd

# Load results
df = pd.read_csv('results/summary_results.csv')

# Key findings
print("=== RQ1: Does structure matter? ===")
print(df.groupby('structure_variant')['rouge1'].mean().sort_values(ascending=False))

print("\n=== Which model is most robust? ===")
for model in df['model'].unique():
    model_df = df[df['model'] == model]
    best = model_df['rouge1'].max()
    worst = model_df['rouge1'].min()
    drop = (best - worst) / best * 100
    print(f"{model}: {drop:.1f}% performance drop")

print("\n=== Best performing combination ===")
print(df.nlargest(1, 'rouge1')[['model', 'structure_variant', 'rouge1', 'rouge2', 'rougeL']])
```

---

## Troubleshooting

### Out of GPU Memory

**Symptoms:** CUDA out of memory error

**Solution 1:** Reduce batch size
```python
# In config.py
BATCH_SIZE = 1  # Default is 4
```

**Solution 2:** Use CPU (slower)
```python
# In experiment.py, line ~60
device = 'cpu'  # Force CPU
```

**Solution 3:** Process fewer samples at a time
```python
runner = ExperimentRunner(n_samples=100)
# Run multiple times with different slices
```

### Model Download Fails

**Symptoms:** Connection timeout, HTTP errors

**Solution:** Pre-download models
```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

models = [
    'facebook/bart-large-cnn',
    'google/pegasus-large',
    'luqh/ClinicalT5-large',
]

for model_name in models:
    print(f"Downloading {model_name}...")
    AutoModelForSeq2SeqLM.from_pretrained(model_name)
    AutoTokenizer.from_pretrained(model_name)
```

### Experiment Crashes Mid-Way

**Symptoms:** Process killed, connection lost

**Good news:** Results are saved incrementally!

**Solution:** Resume by running specific configurations
```python
# Check what's already done
df = pd.read_csv('results/summary_results.csv')
print(df[['model', 'structure_variant']])

# Run only missing configurations
runner = ExperimentRunner()
runner.run_full_experiment(
    models_to_test=['clinical-t5'],  # Only missing model
    variants_to_test=['shuffled']     # Only missing variant
)
```

### BERTScore Takes Forever

**Symptoms:** Evaluation stuck at "Computing BERTScore"

**Solution:** Disable BERTScore
```python
# In experiment.py, line ~100
include_bertscore=False
```

You can compute it separately later on a subset.

---

## Expected Results

Based on similar research, you should see:

### H1: Structure Impact
- **Original:** ROUGE-1 ≈ 0.35-0.45
- **No Headers:** ROUGE-1 ≈ 0.32-0.42 (5-10% drop)
- **No Formatting:** ROUGE-1 ≈ 0.28-0.38 (15-20% drop)
- **Shuffled:** ROUGE-1 ≈ 0.25-0.35 (20-30% drop)

### H2: Model Comparison
- General models (BART-CNN): Larger performance drop (~25%)
- Medical models (Clinical-T5): Smaller performance drop (~15%)

### H3: Key Findings
- Section headers are critical
- Medical models more robust to structure loss
- Shuffling disrupts understanding significantly

---

## Timeline for Thesis

### Week 1: Setup & Pilot
- Day 1-2: Setup, verification, quick tests
- Day 3-4: Pilot study (100 samples)
- Day 5: Review pilot results, adjust if needed

### Week 2: Full Experiment
- Day 1: Start full experiment (overnight run)
- Day 2-3: Generate visualizations and analysis
- Day 4-5: Statistical tests, interpret results

### Week 3: Writing
- Document methodology
- Create results tables and figures
- Write discussion of findings

---

## Next Steps After This RQ

Once RQ1 is complete, you might explore:

- **RQ2:** Effect of note length on summarization
- **RQ3:** Domain-specific fine-tuning on MIMIC
- **RQ4:** Error analysis - what types of clinical info are lost?

---

## Getting Help

If you encounter issues:

1. Check `results/detailed_results.json` for error messages
2. Run `python test_setup.py` to verify setup
3. Try with smaller sample size first (n_samples=10)
4. Check GPU utilization: `nvidia-smi`
5. Review log files in `results/`

Good luck with your thesis! 🎓
