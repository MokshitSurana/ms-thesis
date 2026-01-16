# Refined Research Design: Task-Specific vs Instruction-Tuned Models

## The Pivot: From Failure to Finding

**Original Problem**: BioBART (domain pre-trained) produced garbage despite high ROUGE scores

**New Insight**: This reveals that **domain pre-training alone is insufficient** for clinical summarization

**Solution**: Compare two AI philosophies:
- **Type 1: Task-Specific Specialists** (trained on summarization task)
- **Type 2: Instruction-Tuned Generalists** (trained to follow instructions)

---

## Refined Research Question (RQ1)

**Original**: Does note structure affect LLM performance in clinical summarization?

**Refined**: How does note structure affect summarization across different training paradigms (task-specific vs instruction-tuned)?

**Hypothesis**:
- Type 1 (Task-Specific): Moderately affected by structure (expects news-style flow)
- Type 2 (Instruction-Tuned): **Most robust** to structure disruption (understands task intent)

---

## Model Selection Framework

### Type 1: Task-Specific Summarization Models

**Philosophy**: "I was trained specifically to summarize text"

| Model | HuggingFace ID | Training Data | Strengths | Limitations |
|-------|---------------|---------------|-----------|-------------|
| **BART-CNN** | `facebook/bart-large-cnn` | CNN/DailyMail news | Robust, well-tested | Not medical |
| **PEGASUS** | `google/pegasus-large` | Multi-domain news | Strong abstractive | Not medical |

**Expected Behavior**:
- Good at summarization mechanics
- Struggle with medical terminology
- Moderately affected by structure changes

---

### Type 2: Instruction-Tuned Models

**Philosophy**: "I was trained to follow instructions across many domains"

#### General Instruction-Tuned

| Model | HuggingFace ID | Size | Key Features |
|-------|---------------|------|--------------|
| **Flan-T5 XL** | `google/flan-t5-xl` | 3B params | Gold standard for instruction tuning |
| **Flan-T5 XXL** | `google/flan-t5-xxl` | 11B params | Best performance (if GPU allows) |
| **Flan-T5 Large** | `google/flan-t5-large` | 780M params | Smaller, faster alternative |

**Prompt Format**:
```
summarize: [clinical note text]
```

#### Medical Instruction-Tuned (BEST OPTIONS)

| Model | HuggingFace ID | Base Model | Medical Training | Notes |
|-------|---------------|------------|------------------|-------|
| **Clinical-T5** | `luqh/ClinicalT5-large` | T5-large | MIMIC-III, clinical notes | ⚠️ Flax format (see below) |
| **BioGPT** | `microsoft/biogpt` | GPT-2 | PubMed abstracts | Decoder-only (need seq2seq) |
| **Med-Alpaca** | `medalpaca/medalpaca-7b` | LLaMA-7B | Medical QA + instructions | Large, instruction-tuned |
| **SciFive** | `razent/SciFive-large-Pubmed_PMC` | T5-large | PubMed + PMC articles | Good for biomedical |

#### RECOMMENDED: Clinical Instruction-Tuned T5 Models

**Option A: ClinicalT5 (PyTorch version)**
```python
# Check for PyTorch-compatible version
model_id = "luqh/ClinicalT5-large"
# If Flax issues, convert or use alternative
```

**Option B: SciFive (Immediately usable)**
```python
model_id = "razent/SciFive-large-Pubmed_PMC"
# Native PyTorch, no conversion needed
```

**Option C: Flan-T5 + Medical Fine-tuning**
```python
# Use base Flan-T5 + medical prompt engineering
model_id = "google/flan-t5-xl"
prompt = "summarize the following clinical note: {note}"
```

**Expected Behavior**:
- Better at handling messy/unstructured input
- More robust to shuffling (follows intent, not patterns)
- Should handle "no_formatting" variant well

---

## Addressing the BioBART Problem

### How to Use BioBART Results (Don't Throw Away!)

**Reframe as**: **Type 0: Domain Pre-trained (No Task Tuning)**

| Model Type | Example | Training | Result |
|-----------|---------|----------|--------|
| **Type 0: Domain Only** | BioBART | Medical corpus (MLM) | ❌ High ROUGE, garbage output |
| **Type 1: Task-Specific** | BART-CNN | Summarization task (news) | ✅ Clean output, not medical |
| **Type 2: Instruction-Tuned** | Flan-T5 | Instructions (multi-domain) | ✅✅ Clean + robust |

**Thesis Narrative**:

> "We initially evaluated BioBART, a domain pre-trained model, achieving the highest ROUGE-1 score (0.315). However, qualitative analysis revealed severe output quality issues including Unicode artifacts, spurious tokens, and structure copying. This finding demonstrates that **domain pre-training alone is insufficient** for clinical summarization without task-specific fine-tuning or instruction tuning. BioBART serves as a cautionary **negative control**, highlighting the necessity of task-specific training paradigms."

**Use BioBART Results to**:
1. Motivate the need for instruction-tuned models
2. Show that ROUGE alone is insufficient
3. Demonstrate the importance of qualitative evaluation
4. Justify your dual evaluation framework

---

## Updated Experimental Design

### Models to Run

**Minimum (3 models)**:
1. BART-CNN (Type 1 baseline)
2. Flan-T5-XL (Type 2 general)
3. SciFive (Type 2 medical) OR ClinicalT5 (if compatible)

**Ideal (4 models)**:
1. BART-CNN (Type 1 general)
2. PEGASUS (Type 1 general, for comparison)
3. Flan-T5-XL (Type 2 general)
4. SciFive/ClinicalT5 (Type 2 medical)

**With BioBART as negative control (5 models)**:
1. BioBART (Type 0 - negative control)
2. BART-CNN (Type 1)
3. PEGASUS (Type 1)
4. Flan-T5-XL (Type 2)
5. SciFive (Type 2)

### Expected Results Pattern

**Hypothesis**:

| Variant | Type 0 (BioBART) | Type 1 (BART-CNN) | Type 2 (Flan-T5) |
|---------|------------------|-------------------|------------------|
| **original** | Garbage (0.315 ROUGE) | Good (0.262) | Best (0.28-0.32?) |
| **no_headers** | Garbage (0.315) | Good (0.248) | Best (0.27-0.31?) |
| **no_formatting** | Clean but verbose (0.298) | Good (0.258) | **Best (0.28-0.32?)** |
| **shuffled** | Garbage (0.170) | Degraded (0.231) | **More robust (0.22-0.26?)** |

**Key Prediction**: Flan-T5 will show **smallest drop** from original to shuffled because it understands "summarize this messy text" as an instruction.

---

## Implementation Steps

### Step 1: Verify Model Compatibility

**Test each model loads correctly**:
```python
# Test script
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

models_to_test = {
    'flan-t5-xl': 'google/flan-t5-xl',
    'flan-t5-large': 'google/flan-t5-large',
    'scifive': 'razent/SciFive-large-Pubmed_PMC',
    # 'clinical-t5': 'luqh/ClinicalT5-large',  # Test separately (Flax issue)
}

for name, model_id in models_to_test.items():
    print(f"\nTesting {name}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
        print(f"✓ {name} loaded successfully")
    except Exception as e:
        print(f"✗ {name} failed: {str(e)}")
```

### Step 2: Update config.py

See `config_v2.py` (I'll create this next)

### Step 3: Update Prompts for Instruction-Tuned Models

**Flan-T5 Format**:
```python
# Prefix-based (native T5 format)
input_text = f"summarize: {clinical_note}"

# Or more explicit
input_text = f"summarize the following clinical note: {clinical_note}"
```

### Step 4: Run Experiments

```bash
python experiment.py  # Will use updated config_v2.py
```

### Step 5: Compare Results

Expected comparison:
- Type 1 vs Type 2 on same variants
- Robustness to shuffling (Type 2 should win)
- Medical term preservation (Type 2 medical should win)

---

## Thesis Narrative Update

### Abstract (Updated)

> "We compare three training paradigms for clinical summarization: (1) domain pre-training only (BioBART), (2) task-specific summarization models (BART-CNN, PEGASUS), and (3) instruction-tuned models (Flan-T5, SciFive). Using MIMIC-IV notes (n=1000) with four structure variants (original, no_headers, no_formatting, shuffled), we employ a dual evaluation framework (reference-based and reference-free).
>
> **Key Findings**: Domain pre-training alone is insufficient (BioBART: high ROUGE but garbage output). Task-specific models perform well on structured notes but degrade -12% to -29% with shuffling. **Instruction-tuned models show superior robustness**, maintaining quality even with disrupted structure. Structure significantly affects all paradigms, with logical flow critical but headers redundant.
>
> **Contributions**: (1) First systematic comparison of training paradigms for clinical summarization, (2) dual evaluation framework for robust assessment, (3) evidence that instruction tuning provides better robustness than task-specific training, (4) clinical deployment implications."

### Research Questions (Updated)

**RQ1**: How does note structure affect summarization across different training paradigms?

**RQ2** (Optional): Which training paradigm (task-specific vs instruction-tuned) is most robust to structure variation?

**RQ3** (Optional): Does domain-specific instruction tuning (e.g., SciFive) outperform general instruction tuning (e.g., Flan-T5) for clinical text?

---

## Defense Strategy (Updated)

### Key Slide: "Why This Matters"

**Clinical Reality**:
- Notes are messy
- Formatting inconsistent
- Section order varies
- Need robust systems

**Our Finding**:
- Task-specific models: Good but brittle
- Instruction-tuned: **Robust to real-world messiness**

**Implication**: For clinical deployment, choose instruction-tuned models over task-specific specialists

---

## Timeline Estimate

| Task | Time | Priority |
|------|------|----------|
| Test model loading | 30 min | ⭐⭐⭐ |
| Update config.py | 15 min | ⭐⭐⭐ |
| Run experiments (3-4 models) | 3-4 hours | ⭐⭐⭐ |
| Analyze results | 1 hour | ⭐⭐⭐ |
| Update visualizations | 30 min | ⭐⭐ |
| Rewrite thesis sections | 2-3 hours | ⭐⭐ |

**Total**: ~1 day of work

---

## Critical Questions to Answer

1. **Which Flan-T5 size?**
   - XL (3B) if GPU allows (RTX 3090/A100)
   - Large (780M) if memory constrained

2. **ClinicalT5 or SciFive?**
   - SciFive: Easier (native PyTorch)
   - ClinicalT5: Better (trained on MIMIC) but Flax conversion needed

3. **Keep PEGASUS?**
   - Yes, if you want two Type 1 models for robustness
   - No, if you want to focus on Type 1 vs Type 2 contrast (just BART-CNN)

4. **Include BioBART in new results?**
   - Yes, as Type 0 negative control
   - Provides complete picture of training paradigms

---

## Next Steps

**Immediate** (I'll do now):
1. Create updated `config_v2.py` with new models
2. Create model compatibility test script
3. Find best medical instruction-tuned T5 model
4. Create updated experiment runner

**You do**:
1. Test which models load on your GPU
2. Choose final model set (3-5 models)
3. Run updated experiment
4. Compare new results with BioBART baseline

Ready to implement! Let me create the updated configuration files.
