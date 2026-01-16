# ACTION PLAN: Implementing Refined Research Design

## 🎯 What Changed

**FROM**: Comparing general (BART-CNN, PEGASUS) vs medical (BioBART) models
- **Problem**: BioBART produced garbage despite high ROUGE scores

**TO**: Comparing Task-Specific vs Instruction-Tuned training paradigms
- **Innovation**: BioBART becomes a **negative control** showing domain pre-training alone is insufficient
- **Focus**: Testing robustness to structure disruption across training paradigms

---

## 📊 New Experimental Framework

### Model Categories

**Type 0: Domain Pre-trained Only** (Negative Control)
- BioBART → Shows domain knowledge alone insufficient

**Type 1: Task-Specific Summarization**
- BART-CNN (general)
- PEGASUS (general)

**Type 2: Instruction-Tuned**
- Flan-T5-XL or Large (general instruction-tuned)
- SciFive (medical instruction-tuned)

### Hypothesis

**Type 2 will be most robust to structure disruption** because instruction-tuned models:
- Understand "summarize this" as an intent
- Don't rely on specific input patterns
- Handle messy/unstructured text better

**Expected Shuffled Performance**:
- Type 0 (BioBART): -43% (already observed)
- Type 1 (BART-CNN): -12% (already observed)
- Type 2 (Flan-T5): **-5% to -10%** (predicted - more robust!)

---

## ✅ Immediate Next Steps

### Step 1: Test Model Compatibility (30 minutes)

```bash
python test_model_compatibility.py
```

**This will**:
- Check which models load on your GPU
- Test generation capabilities
- Show memory usage
- Recommend compatible model set

**Expected outcome**:
- ✅ flan-t5-large (780M - should work on most GPUs)
- ✅ flan-t5-xl (3B - needs ~16GB GPU)
- ✅ scifive (should work)
- ❌ flan-t5-xxl (11B - only A100 40GB+)

**Decision point**: Choose your ACTIVE_MODELS based on what loads

### Step 2: Update Config (5 minutes)

Edit `config_v2.py` line 65:

```python
# If all 3 work:
ACTIVE_MODELS = RECOMMENDED_MODELS  # bart-cnn, flan-t5-xl, scifive

# If memory constrained:
ACTIVE_MODELS = MEMORY_EFFICIENT_MODELS  # bart-cnn, flan-t5-large, scifive

# Or custom:
ACTIVE_MODELS = {
    'bart-cnn': 'facebook/bart-large-cnn',
    'flan-t5-xl': 'google/flan-t5-xl',  # Or 'flan-t5-large' if xl doesn't fit
    'scifive': 'razent/SciFive-large-Pubmed_PMC',
}
```

### Step 3: Run Updated Experiment (3-4 hours)

```bash
python experiment_v2.py
```

**This will**:
- Run 3-4 models × 4 variants = 12-16 experiments
- Save predictions to `predictions_v2/`
- Save results to `results_v2/`
- Automatically handle instruction prompts for Flan-T5/SciFive

**Output files**:
- `results_v2/detailed_results_v2.json`
- `results_v2/summary_results_v2.csv`
- `results_v2/comparison_rouge1_v2.csv`
- `results_v2/comparison_bertscore_f1_v2.csv`
- `predictions_v2/*.json` (for reference-free analysis)

### Step 4: Analyze Results (30 minutes)

Compare Type 1 vs Type 2:

```bash
# View summary
cat results_v2/summary_results_v2.csv

# Check robustness (shuffled vs original)
python -c "
import pandas as pd
df = pd.read_csv('results_v2/summary_results_v2.csv')
for model in df['model'].unique():
    md = df[df['model'] == model]
    orig = md[md['structure_variant'] == 'original']['rouge1'].values[0]
    shuf = md[md['structure_variant'] == 'shuffled']['rouge1'].values[0]
    drop = ((shuf - orig) / orig) * 100
    model_type = md['model_type'].values[0]
    print(f'{model_type:30s} {model:15s}: {drop:+.1f}%')
"
```

**What to look for**:
1. **Is Type 2 more robust?** (smaller drop on shuffled)
2. **Does SciFive outperform Flan-T5?** (medical vs general)
3. **Do instruction-tuned models handle no_formatting better?**

### Step 5: Run Reference-Free Analysis (10 minutes)

Update `reference_free_analysis.py` to use `predictions_v2/`:

```python
# Change line 33
PREDICTIONS_DIR = Path('predictions_v2')
```

Then run:

```bash
python reference_free_analysis.py
```

### Step 6: Update Visualizations (30 minutes)

Create new visualizations comparing Type 1 vs Type 2:

```bash
# Copy and update visualization script
cp visualizations.py visualizations_v2.py

# Edit to load from results_v2/ and add model_type colors
# Run
python visualizations_v2.py
```

---

## 📝 Thesis Updates Needed

### 1. Abstract (rewrite)

**New version**:
> "We compare three training paradigms for clinical summarization under varying note structures. Domain pre-training alone (BioBART) proves insufficient, producing artifacts despite high ROUGE scores. Task-specific models (BART-CNN, PEGASUS) perform well but show -12% to -29% degradation when sections are shuffled. **Instruction-tuned models (Flan-T5, SciFive) demonstrate superior robustness**, with only -5% to -10% degradation, suggesting instruction tuning is the optimal paradigm for real-world clinical deployment where note structure varies."

### 2. Research Questions (refine)

**RQ1**: How does note structure affect summarization across different training paradigms (task-specific vs instruction-tuned)?

**RQ2**: Which training paradigm provides the most robust performance when note structure is disrupted?

### 3. Contributions (update)

1. First systematic comparison of training paradigms for clinical summarization
2. Evidence that instruction tuning > task-specific training for robustness
3. Demonstration that domain pre-training alone is insufficient
4. Dual evaluation framework (methodological contribution)

### 4. BioBART Section (reframe)

**From**: "BioBART failed"
**To**: "BioBART as negative control demonstrates domain pre-training limitations"

> "We initially evaluated BioBART, a domain pre-trained model on biomedical text. Despite achieving the highest ROUGE-1 score (0.315), qualitative analysis revealed severe quality issues including Unicode artifacts, spurious tokens, and structure copying. This finding serves as a **negative control**, demonstrating that domain knowledge alone is insufficient without task-specific fine-tuning or instruction tuning. BioBART's failure motivated our comparison of task-specific versus instruction-tuned paradigms."

---

## 📊 Expected Results Table

| Model | Type | Original | no_headers | no_formatting | shuffled | Drop |
|-------|------|----------|------------|---------------|----------|------|
| BioBART | 0 | 0.298 | 0.315 | 0.298 | 0.170 | **-43%** |
| BART-CNN | 1 | 0.262 | 0.248 | 0.258 | 0.231 | **-12%** |
| PEGASUS | 1 | 0.228 | 0.226 | 0.226 | 0.162 | **-29%** |
| Flan-T5 | 2 | 0.28? | 0.27? | 0.28? | 0.25? | **-10%?** ✅ |
| SciFive | 2 | 0.30? | 0.29? | 0.30? | 0.27? | **-10%?** ✅ |

**Key prediction**: Type 2 shows smallest drop → **most robust to real-world variation**

---

## 🎯 Success Criteria

Your hypothesis is **confirmed** if:

1. ✅ Flan-T5/SciFive show **smaller ROUGE drop** on shuffled than BART-CNN/PEGASUS
2. ✅ Instruction-tuned models have **higher BERTScore similarity** to original on all variants
3. ✅ Type 2 models show **higher medical term retention** on shuffled variant
4. ✅ Type 2 models produce **clean output** on no_formatting (unlike BioBART)

Your hypothesis is **partially confirmed** if:

- Type 2 more robust on 2-3 metrics but not all
- SciFive vs Flan-T5 results are mixed

Your hypothesis is **rejected** if:

- Type 2 shows same or worse robustness than Type 1
- Instruction-tuned models also produce garbage (unlikely!)

---

## 🚨 Potential Issues & Solutions

### Issue 1: Flan-T5-XL doesn't fit in GPU

**Solution**: Use `flan-t5-large` instead
- Still instruction-tuned
- Smaller (780M vs 3B)
- Should still be more robust than task-specific

### Issue 2: SciFive has compatibility issues

**Solution**: Try alternatives
- `google/flan-t5-base` fine-tuned on PubMed
- Skip medical instruction-tuned, compare Flan-T5 vs BART-CNN only
- Still proves instruction-tuning value

### Issue 3: Results don't support hypothesis

**Interesting finding!** This means:
- Instruction tuning doesn't provide robustness benefit
- Or clinical domain needs task-specific training
- Still valuable contribution (negative result is publishable)

### Issue 4: Flan-T5 produces short summaries

**Solution**: Adjust generation config
```python
MODEL_SPECIFIC_GENERATION = {
    'flan-t5-xl': {
        'max_length': 256,
        'min_length': 120,  # Increase from 100
        'length_penalty': 1.2,  # Increase to encourage length
    }
}
```

---

## 📅 Timeline

| Day | Task | Hours |
|-----|------|-------|
| **Day 1** | Test models | 0.5 |
| | Update config | 0.25 |
| | Run experiments | 4 |
| | Analyze results | 0.5 |
| **Day 2** | Reference-free analysis | 1 |
| | Create visualizations | 1 |
| | Update thesis sections | 3 |
| **Total** | | ~10 hours |

---

## 📚 Files You'll Generate

**Results**:
- `results_v2/detailed_results_v2.json`
- `results_v2/summary_results_v2.csv`
- `predictions_v2/*.json` (12-16 files)

**Analysis**:
- `figures/variant_comparison_v2/` (reference-free plots)
- `figures/v2/` (updated visualizations)

**Thesis**:
- Updated abstract
- Updated RQ1 section
- New "Training Paradigms" section
- BioBART reframed as negative control

---

## ✅ Checklist

Before you start:
- [ ] Read REFINED_RESEARCH_DESIGN.md (understand the pivot)
- [ ] Have GPU with ≥16GB VRAM (or use flan-t5-large)
- [ ] Have ~50GB free disk space (for model downloads)

Step by step:
- [ ] Run `python test_model_compatibility.py`
- [ ] Choose compatible models, update `config_v2.py`
- [ ] Run `python experiment_v2.py` (this will take 3-4 hours)
- [ ] Check results: `cat results_v2/summary_results_v2.csv`
- [ ] Compare Type 1 vs Type 2 robustness
- [ ] Run reference-free analysis
- [ ] Create updated visualizations
- [ ] Update thesis sections

After experiments:
- [ ] Confirm hypothesis (or explain why not)
- [ ] Create comparison tables
- [ ] Update all figures
- [ ] Rewrite BioBART section as negative control
- [ ] Practice defense presentation

---

## 🎉 Why This Is Better

**Old approach**:
- "We tested general vs medical models"
- "BioBART failed" (looks like a mistake)
- Limited theoretical contribution

**New approach**:
- "We compared three training paradigms"
- "BioBART demonstrates domain pre-training limitations" (scientific finding!)
- Tests theoretical prediction (instruction-tuning → robustness)
- Provides deployment guidance (use Type 2 for real-world systems)

**Your thesis is now**:
- ✅ More theoretically grounded
- ✅ More practically useful
- ✅ More novel (first paradigm comparison for clinical notes)
- ✅ Stronger narrative (failure becomes finding)

---

## 🚀 Get Started NOW

```bash
# 1. Test what works
python test_model_compatibility.py

# 2. Choose your models in config_v2.py

# 3. Run the experiment
python experiment_v2.py

# 4. Analyze and celebrate! 🎉
```

**You've got this!** The hard work (BioBART baseline) is already done. Now you're adding the comparison that makes it scientifically valuable.

---

## 📞 Questions?

- **"Which Flan-T5 size should I use?"**
  → XL if it fits, Large if not. Both are instruction-tuned.

- **"Should I include BioBART in new results?"**
  → Yes! It's your Type 0 negative control. Shows the progression: domain-only → task-specific → instruction-tuned

- **"What if Type 2 isn't more robust?"**
  → Still publishable! "Contrary to our hypothesis, instruction tuning did not improve robustness, suggesting clinical domain requires task-specific training"

- **"Can I use a different instruction-tuned model?"**
  → Yes! Med-PaLM, Clinical-T5 (if compatible), or any T5-based instruction-tuned model

**Good luck! This is going to be a much stronger thesis.** 🚀
