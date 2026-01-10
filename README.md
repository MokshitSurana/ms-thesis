# RQ1: Effect of Note Structure on Clinical Summarization

## Research Question

**How does the structural organization of clinical notes (e.g., use of section headers, bullet lists, and consistent formatting) influence the performance of large language models on clinical NLP tasks such as summarization?**

## Overview

This repository implements a comprehensive experimental framework to investigate how clinical note structure affects LLM summarization performance.

### Experimental Design

**Models Tested:**
- **General Models:** BART-CNN, PEGASUS
- **Medical Models:** Clinical-T5, BioGPT

**Structure Variants:**
1. **Original** - Maintains original structure with section headers (Chief Complaint, HPI, etc.)
2. **No Headers** - Removes section headers but keeps content organization
3. **No Formatting** - Removes all formatting (headers, bullets, newlines) → continuous text
4. **Shuffled** - Randomly shuffles section order to disrupt logical flow

**Evaluation Metrics:**
- ROUGE-1, ROUGE-2, ROUGE-L
- BERTScore (F1, Precision, Recall)
- Length statistics

## Project Structure

```
ms-thesis/
├── config.py                    # Configuration (models, parameters)
├── structure_manipulation.py    # Note structure transformation
├── models.py                    # Model wrappers
├── evaluation.py                # Evaluation metrics
├── experiment.py                # Main experiment runner
├── visualize.py                 # Results visualization
├── requirements.txt             # Dependencies
└── results/                     # Output directory
    ├── summary_results.csv      # Main results table
    ├── detailed_results.json    # Full results with samples
    └── figures/                 # Generated plots
```

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download NLTK Data (for ROUGE)

```python
python -c "import nltk; nltk.download('punkt')"
```

### 3. Prepare Data

Place your MIMIC-IV dataset as `mimic_1000_notes_with_references.csv` with columns:
- `note_id`
- `text` (full clinical note)
- `reference_summary` (gold standard summary)

## Usage

### Quick Test (10 samples)

Test the pipeline with a small subset:

```bash
python experiment.py
```

This runs BART-CNN on 2 structure variants (original, no_formatting) with 10 samples.

### Full Experiment

Run complete experiment with all models and variants:

```bash
python experiment.py --full
```

This will:
1. Process all 1000 notes
2. Test all models × all structure variants
3. Save results to `results/`

**Note:** Full experiment may take several hours depending on GPU.

### Custom Experiment

```python
from experiment import ExperimentRunner

runner = ExperimentRunner(n_samples=100)

# Test specific configurations
runner.run_full_experiment(
    models_to_test=['bart-cnn', 'clinical-t5'],
    variants_to_test=['original', 'no_formatting']
)

# Create comparison tables
runner.create_comparison_table()
```

## Analyzing Results

### Generate Visualizations

```bash
python visualize.py
```

This creates:
- Bar charts comparing models across structures
- Heatmaps showing performance patterns
- Line plots showing structure degradation effects
- Statistical summaries

### Manual Analysis

```python
import pandas as pd

# Load results
results = pd.read_csv('results/summary_results.csv')

# Compare models
print(results.groupby('model')['rouge1'].mean())

# Compare structures
print(results.groupby('structure_variant')['rouge1'].mean())

# Best combination
print(results.nlargest(1, 'rouge1'))
```

## Key Research Hypotheses

### H1: Structure Matters
Models will perform better on structured notes (original) compared to unstructured notes (no_formatting).

**Expected:** `original > no_headers > no_formatting`

### H2: Medical Models More Robust
Medical models (Clinical-T5) will be more robust to structure degradation than general models (BART-CNN).

**Expected:** Smaller performance drop for Clinical-T5 when structure is removed.

### H3: Shuffling Disrupts Understanding
Shuffled section order will significantly hurt performance, showing models rely on conventional note organization.

**Expected:** `shuffled` will have lowest scores.

## Results Interpretation

### Key Metrics

- **ROUGE-1**: Unigram overlap (captures key terms)
- **ROUGE-2**: Bigram overlap (captures phrasing)
- **ROUGE-L**: Longest common subsequence (captures structure)
- **BERTScore**: Semantic similarity

### What to Look For

1. **Performance Drop**: How much does removing structure hurt?
   ```
   Drop% = (original_score - no_formatting_score) / original_score × 100
   ```

2. **Model Comparison**: Do medical models handle unstructured notes better?

3. **Critical Structure Elements**: Which matters more - headers or formatting?

## Extending the Research

### Add More Models

Edit `config.py`:

```python
MODELS = {
    'general': {
        'bart-cnn': 'facebook/bart-large-cnn',
        'pegasus': 'google/pegasus-large',
        't5-base': 't5-base',  # Add this
    },
    'medical': {
        'clinical-t5': 'luqh/ClinicalT5-large',
        'your-model': 'your/model-name',  # Add this
    }
}
```

### Add More Structure Variants

Implement in `structure_manipulation.py`:

```python
def bullets_only(self, text: str) -> str:
    """Keep only bullet points."""
    # Your implementation
    pass
```

Then add to `config.py`:
```python
STRUCTURE_VARIANTS = [..., 'bullets_only']
```

### Add Custom Metrics

Implement in `evaluation.py`:

```python
def compute_medical_accuracy(self, predictions, references):
    """Custom medical metric."""
    # Your implementation
    pass
```

## Troubleshooting

### Out of Memory (GPU)

Reduce batch size in `config.py`:
```python
BATCH_SIZE = 1  # Default is 4
```

### Slow Evaluation

Disable BERTScore in `experiment.py`:
```python
metrics = self.evaluator.compute_all_metrics(
    predictions, references,
    include_bertscore=False  # Set to False
)
```

### Model Download Issues

If HuggingFace downloads fail, manually download models first:
```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model = AutoModelForSeq2SeqLM.from_pretrained('facebook/bart-large-cnn')
tokenizer = AutoTokenizer.from_pretrained('facebook/bart-large-cnn')
```

## Citation

If you use this framework, please cite:

```bibtex
@mastersthesis{yourname2024structure,
  title={Effect of Note Structure on Large Language Model Performance in Clinical Summarization},
  author={Your Name},
  year={2024},
  school={Your University}
}
```

## License

This research framework is provided for academic use.

## Contact

For questions or collaboration: your.email@example.com
