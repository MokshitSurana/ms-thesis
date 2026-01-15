"""
Comprehensive Reference-Free Analysis
Calculates all relevant metrics comparing variants to original baseline.

Metrics:
1. BERTScore (semantic similarity) - already done by ref_free.py
2. Length statistics (verbosity)
3. Medical term overlap (content preservation)
4. N-gram overlap (lexical similarity)
5. Sentence structure similarity
6. Extractiveness (copy-paste behavior)
7. Lexical diversity
8. Novel content generation
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter
import re
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.util import ngrams as nltk_ngrams
import matplotlib.pyplot as plt
import seaborn as sns

# Setup
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("colorblind")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 11
plt.rcParams['font.family'] = 'serif'

OUTPUT_DIR = Path('figures/comprehensive_reference_free')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Medical term patterns
MEDICAL_TERMS = [
    r'\b(?:diagnosis|diagnosed|diagnose)\b',
    r'\b(?:treatment|treated|therapy)\b',
    r'\b(?:patient|pt)\b',
    r'\b(?:symptom|symptoms)\b',
    r'\b(?:procedure|surgery|surgical)\b',
    r'\b(?:medication|medications|med|meds)\b',
    r'\b(?:condition|conditions)\b',
    r'\b(?:admitted|admission|discharge|discharged)\b',
    r'\b(?:history|hx)\b',
    r'\b(?:examination|exam|examined)\b',
]


def load_results_from_json():
    """Load predictions from detailed_results JSON."""
    results_file = Path('./results/detailed_results_with_bertscore.json')

    if not results_file.exists():
        print(f"Error: {results_file} not found")
        print("Run experiment.py first to generate results")
        return None

    with open(results_file, 'r') as f:
        data = json.load(f)

    # Organize by model and variant
    organized = {}
    for item in data:
        model = item['model']
        variant = item['structure_variant']

        if model not in organized:
            organized[model] = {}
        if variant not in organized[model]:
            organized[model][variant] = {
                'predictions': [],
                'references': [],
                'inputs': []
            }

        # Get predictions (sample up to 10 shown in results)
        if 'predictions' in item and item['predictions']:
            organized[model][variant]['predictions'].extend(item['predictions'])

        # If we don't have full predictions, we'll need to regenerate
        # For now, mark as incomplete
        if len(organized[model][variant]['predictions']) == 0:
            organized[model][variant]['incomplete'] = True

    return organized


# ============================================================================
# METRIC 1: Length Statistics
# ============================================================================

def calculate_length_metrics(original_preds, variant_preds):
    """Calculate comprehensive length statistics."""
    orig_lengths = [len(p.split()) for p in original_preds]
    var_lengths = [len(p.split()) for p in variant_preds]

    return {
        'original_mean': np.mean(orig_lengths),
        'original_std': np.std(orig_lengths),
        'variant_mean': np.mean(var_lengths),
        'variant_std': np.std(var_lengths),
        'mean_change': np.mean(var_lengths) - np.mean(orig_lengths),
        'pct_change': ((np.mean(var_lengths) - np.mean(orig_lengths)) / np.mean(orig_lengths)) * 100,
        'correlation': np.corrcoef(orig_lengths, var_lengths)[0, 1]
    }


# ============================================================================
# METRIC 2: Medical Term Overlap
# ============================================================================

def extract_medical_terms(text):
    """Extract medical terms from text."""
    terms = set()
    text_lower = text.lower()

    for pattern in MEDICAL_TERMS:
        matches = re.findall(pattern, text_lower)
        terms.update(matches)

    return terms


def calculate_medical_term_metrics(original_preds, variant_preds):
    """Calculate medical term overlap metrics."""
    jaccard_scores = []
    retention_scores = []

    for orig, var in zip(original_preds, variant_preds):
        orig_terms = extract_medical_terms(orig)
        var_terms = extract_medical_terms(var)

        # Jaccard similarity
        if len(orig_terms | var_terms) > 0:
            jaccard = len(orig_terms & var_terms) / len(orig_terms | var_terms)
            jaccard_scores.append(jaccard)

        # Retention (what % of original terms are kept)
        if len(orig_terms) > 0:
            retention = len(orig_terms & var_terms) / len(orig_terms)
            retention_scores.append(retention)

    return {
        'jaccard_mean': np.mean(jaccard_scores) if jaccard_scores else 0,
        'jaccard_std': np.std(jaccard_scores) if jaccard_scores else 0,
        'retention_mean': np.mean(retention_scores) if retention_scores else 0,
        'retention_std': np.std(retention_scores) if retention_scores else 0
    }


# ============================================================================
# METRIC 3: N-gram Overlap
# ============================================================================

def extract_ngrams(text, n):
    """Extract n-grams from text."""
    tokens = word_tokenize(text.lower())
    return list(nltk_ngrams(tokens, n))


def calculate_ngram_overlap(original_preds, variant_preds, n=1):
    """Calculate n-gram overlap (Jaccard similarity)."""
    overlaps = []

    for orig, var in zip(original_preds, variant_preds):
        orig_ngrams = set(extract_ngrams(orig, n))
        var_ngrams = set(extract_ngrams(var, n))

        if len(orig_ngrams | var_ngrams) > 0:
            jaccard = len(orig_ngrams & var_ngrams) / len(orig_ngrams | var_ngrams)
            overlaps.append(jaccard)

    return {
        f'{n}gram_overlap_mean': np.mean(overlaps) if overlaps else 0,
        f'{n}gram_overlap_std': np.std(overlaps) if overlaps else 0
    }


# ============================================================================
# METRIC 4: Sentence Structure
# ============================================================================

def calculate_sentence_metrics(original_preds, variant_preds):
    """Compare sentence-level statistics."""
    orig_sent_counts = [len(sent_tokenize(p)) for p in original_preds]
    var_sent_counts = [len(sent_tokenize(p)) for p in variant_preds]

    orig_sent_lengths = []
    var_sent_lengths = []

    for pred in original_preds:
        sents = sent_tokenize(pred)
        if sents:
            orig_sent_lengths.extend([len(s.split()) for s in sents])

    for pred in variant_preds:
        sents = sent_tokenize(pred)
        if sents:
            var_sent_lengths.extend([len(s.split()) for s in sents])

    return {
        'sent_count_orig': np.mean(orig_sent_counts),
        'sent_count_var': np.mean(var_sent_counts),
        'sent_count_change': np.mean(var_sent_counts) - np.mean(orig_sent_counts),
        'sent_length_orig': np.mean(orig_sent_lengths) if orig_sent_lengths else 0,
        'sent_length_var': np.mean(var_sent_lengths) if var_sent_lengths else 0,
        'sent_count_correlation': np.corrcoef(orig_sent_counts, var_sent_counts)[0, 1]
    }


# ============================================================================
# METRIC 5: Lexical Diversity
# ============================================================================

def calculate_lexical_diversity(text):
    """Calculate type-token ratio (unique words / total words)."""
    tokens = word_tokenize(text.lower())
    if len(tokens) == 0:
        return 0
    return len(set(tokens)) / len(tokens)


def calculate_diversity_metrics(original_preds, variant_preds):
    """Compare lexical diversity."""
    orig_diversity = [calculate_lexical_diversity(p) for p in original_preds]
    var_diversity = [calculate_lexical_diversity(p) for p in variant_preds]

    return {
        'diversity_orig': np.mean(orig_diversity),
        'diversity_var': np.mean(var_diversity),
        'diversity_change': np.mean(var_diversity) - np.mean(orig_diversity),
        'diversity_correlation': np.corrcoef(orig_diversity, var_diversity)[0, 1]
    }


# ============================================================================
# METRIC 6: Novel Content (vs Input)
# ============================================================================

def calculate_novel_content(predictions, inputs):
    """
    Calculate what % of prediction is NOT in the input (abstractiveness).
    Higher = more abstractive, Lower = more extractive
    """
    novelty_scores = []

    for pred, inp in zip(predictions, inputs):
        pred_words = set(word_tokenize(pred.lower()))
        input_words = set(word_tokenize(inp.lower()))

        if len(pred_words) > 0:
            novel_words = pred_words - input_words
            novelty = len(novel_words) / len(pred_words)
            novelty_scores.append(novelty)

    return {
        'novelty_mean': np.mean(novelty_scores) if novelty_scores else 0,
        'novelty_std': np.std(novelty_scores) if novelty_scores else 0
    }


# ============================================================================
# Main Analysis
# ============================================================================

def analyze_all_metrics(model_name, original_data, variant_data, variant_name):
    """Run all metric calculations."""
    print(f"\n  Analyzing: {model_name} - {variant_name}")

    orig_preds = original_data['predictions']
    var_preds = variant_data['predictions']

    if not orig_preds or not var_preds:
        print(f"    ✗ No predictions available")
        return None

    # Ensure same length
    min_len = min(len(orig_preds), len(var_preds))
    orig_preds = orig_preds[:min_len]
    var_preds = var_preds[:min_len]

    print(f"    Sample size: {min_len}")

    results = {
        'model': model_name,
        'variant': variant_name,
        'n_samples': min_len
    }

    # Calculate all metrics
    print("    - Length statistics...")
    results.update(calculate_length_metrics(orig_preds, var_preds))

    print("    - Medical term overlap...")
    results.update(calculate_medical_term_metrics(orig_preds, var_preds))

    print("    - N-gram overlap (1,2,3)...")
    for n in [1, 2, 3]:
        results.update(calculate_ngram_overlap(orig_preds, var_preds, n))

    print("    - Sentence structure...")
    results.update(calculate_sentence_metrics(orig_preds, var_preds))

    print("    - Lexical diversity...")
    results.update(calculate_diversity_metrics(orig_preds, var_preds))

    # If we have inputs, calculate novelty
    if 'inputs' in original_data and original_data['inputs']:
        print("    - Novel content...")
        orig_inputs = original_data['inputs'][:min_len]
        var_inputs = variant_data['inputs'][:min_len]

        results['orig_novelty'] = calculate_novel_content(orig_preds, orig_inputs)['novelty_mean']
        results['var_novelty'] = calculate_novel_content(var_preds, var_inputs)['novelty_mean']

    print(f"    ✓ Complete")

    return results


def main():
    print("\n" + "="*70)
    print("Comprehensive Reference-Free Analysis")
    print("="*70)

    # Load data
    print("\nLoading predictions from results...")
    data = load_results_from_json()

    if not data:
        return

    all_results = []

    models = ['bart-cnn', 'pegasus', 'biobart']
    variants = ['no_headers', 'no_formatting', 'shuffled']

    for model in models:
        if model not in data:
            print(f"\n✗ No data for {model}")
            continue

        if 'original' not in data[model]:
            print(f"\n✗ No original predictions for {model}")
            continue

        print(f"\n{'='*60}")
        print(f"Model: {model}")
        print('='*60)

        original_data = data[model]['original']

        for variant in variants:
            if variant not in data[model]:
                print(f"  ✗ No data for {variant}")
                continue

            variant_data = data[model][variant]

            result = analyze_all_metrics(model, original_data, variant_data, variant)
            if result:
                all_results.append(result)

    if not all_results:
        print("\n✗ No results generated")
        print("\nNote: This script needs full predictions, not just samples.")
        print("You may need to modify experiment.py to save all predictions.")
        return

    # Create dataframe
    df = pd.DataFrame(all_results)

    # Save results
    df.to_csv(OUTPUT_DIR / 'comprehensive_metrics.csv', index=False)
    print(f"\n✓ Saved: {OUTPUT_DIR / 'comprehensive_metrics.csv'}")

    # Create summary table
    print("\n" + "="*70)
    print("Summary of Key Metrics:")
    print("="*70)

    summary_cols = ['model', 'variant', 'pct_change', 'retention_mean',
                    '1gram_overlap_mean', '2gram_overlap_mean',
                    'diversity_change', 'sent_count_change']

    if all(col in df.columns for col in summary_cols):
        print(df[summary_cols].to_string(index=False))

    print("\n" + "="*70)
    print("Interpretation Guide:")
    print("="*70)
    print("- pct_change: Length change from original (negative = shorter)")
    print("- retention_mean: Medical term retention (1.0 = perfect retention)")
    print("- Ngram_overlap: Lexical similarity (1.0 = identical)")
    print("- diversity_change: Change in lexical diversity (type-token ratio)")
    print("- sent_count_change: Change in number of sentences")
    print("="*70 + "\n")


if __name__ == '__main__':
    # Download NLTK data if needed
    import nltk
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("Downloading NLTK punkt tokenizer...")
        nltk.download('punkt')

    main()
