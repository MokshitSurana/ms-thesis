"""
Reference-Free Analysis: Compare Structure Variants Against Original Baseline
Shows how much each structure manipulation changes the model's output.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter
import re
import matplotlib.pyplot as plt
import seaborn as sns

# Try to import bert_score, but make it optional
try:
    from bert_score import score as bert_score
    BERTSCORE_AVAILABLE = True
except ImportError:
    BERTSCORE_AVAILABLE = False
    print("Warning: bert_score not installed. BERTScore analysis will be skipped.")
    print("Install with: pip install bert-score")

# Set style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("colorblind")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 11
plt.rcParams['font.family'] = 'serif'

# Directories
PREDICTIONS_DIR = Path('predictions')
OUTPUT_DIR = Path('figures/reference_free')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Medical term patterns (common clinical concepts)
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


def load_predictions(model_name, structure_variant):
    """Load predictions from JSON file."""
    file_path = PREDICTIONS_DIR / f'{model_name}_{structure_variant}_predictions.json'

    if not file_path.exists():
        print(f"Warning: {file_path} not found")
        return []

    with open(file_path, 'r') as f:
        data = json.load(f)

    return [item['prediction'] for item in data]


def calculate_bertscore_between_variants(predictions_original, predictions_variant):
    """Calculate BERTScore between original and variant predictions."""
    if not BERTSCORE_AVAILABLE:
        return {
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0
        }

    if not predictions_original or not predictions_variant:
        return None

    # Limit to first 100 for speed (representative sample)
    sample_size = min(100, len(predictions_original), len(predictions_variant))

    P, R, F1 = bert_score(
        predictions_variant[:sample_size],
        predictions_original[:sample_size],
        lang='en',
        model_type='microsoft/deberta-xlarge-mnli',
        verbose=False
    )

    return {
        'precision': P.mean().item(),
        'recall': R.mean().item(),
        'f1': F1.mean().item()
    }


def extract_medical_terms(text):
    """Extract medical terms from text."""
    terms = set()
    text_lower = text.lower()

    for pattern in MEDICAL_TERMS:
        matches = re.findall(pattern, text_lower)
        terms.update(matches)

    return terms


def calculate_term_overlap(predictions_original, predictions_variant):
    """Calculate overlap of medical terms between variants."""
    overlaps = []

    for orig, var in zip(predictions_original, predictions_variant):
        terms_orig = extract_medical_terms(orig)
        terms_var = extract_medical_terms(var)

        if len(terms_orig) > 0:
            overlap = len(terms_orig & terms_var) / len(terms_orig)
        else:
            overlap = 0.0

        overlaps.append(overlap)

    return np.mean(overlaps) if overlaps else 0.0


def calculate_length_stats(predictions_original, predictions_variant):
    """Calculate length statistics."""
    lengths_orig = [len(p.split()) for p in predictions_original]
    lengths_var = [len(p.split()) for p in predictions_variant]

    return {
        'original_mean': np.mean(lengths_orig),
        'variant_mean': np.mean(lengths_var),
        'length_change_pct': ((np.mean(lengths_var) - np.mean(lengths_orig)) / np.mean(lengths_orig)) * 100,
        'original_std': np.std(lengths_orig),
        'variant_std': np.std(lengths_var)
    }


def analyze_model(model_name):
    """Analyze all variants for a given model against original baseline."""
    print(f"\nAnalyzing {model_name}...")

    # Load original predictions (baseline)
    predictions_original = load_predictions(model_name, 'original')

    if not predictions_original:
        print(f"  ✗ No original predictions found for {model_name}")
        return None

    print(f"  ✓ Loaded {len(predictions_original)} original predictions")

    results = {}

    for variant in ['no_headers', 'no_formatting', 'shuffled']:
        print(f"  Comparing against {variant}...")

        predictions_variant = load_predictions(model_name, variant)

        if not predictions_variant:
            continue

        # Calculate BERTScore (semantic similarity)
        if BERTSCORE_AVAILABLE:
            print(f"    - Computing BERTScore (sample n=100)...")
        else:
            print(f"    - Skipping BERTScore (not installed)...")
        bert_scores = calculate_bertscore_between_variants(
            predictions_original, predictions_variant
        )

        # Calculate length statistics
        print(f"    - Computing length statistics...")
        length_stats = calculate_length_stats(
            predictions_original, predictions_variant
        )

        # Calculate medical term overlap
        print(f"    - Computing medical term overlap...")
        term_overlap = calculate_term_overlap(
            predictions_original, predictions_variant
        )

        results[variant] = {
            'bertscore': bert_scores,
            'length_stats': length_stats,
            'term_overlap': term_overlap
        }

        if BERTSCORE_AVAILABLE:
            print(f"    ✓ BERTScore F1: {bert_scores['f1']:.4f}")
        print(f"    ✓ Length change: {length_stats['length_change_pct']:+.1f}%")
        print(f"    ✓ Term overlap: {term_overlap:.2%}")

    return results


def create_comparison_dataframe(all_results):
    """Create a comprehensive comparison dataframe."""
    rows = []

    for model, variants in all_results.items():
        for variant, metrics in variants.items():
            rows.append({
                'Model': model,
                'Variant': variant,
                'BERTScore_F1': metrics['bertscore']['f1'],
                'BERTScore_Precision': metrics['bertscore']['precision'],
                'BERTScore_Recall': metrics['bertscore']['recall'],
                'Length_Change_Pct': metrics['length_stats']['length_change_pct'],
                'Original_Length': metrics['length_stats']['original_mean'],
                'Variant_Length': metrics['length_stats']['variant_mean'],
                'Medical_Term_Overlap': metrics['term_overlap']
            })

    return pd.DataFrame(rows)


def plot_bertscore_deviation(df):
    """
    Plot how much each variant deviates from original (semantic similarity).
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    variant_order = ['no_headers', 'no_formatting', 'shuffled']
    variant_labels = ['No Headers', 'No Formatting', 'Shuffled']

    models = ['bart-cnn', 'pegasus', 'biobart']
    model_labels = ['BART-CNN', 'PEGASUS', 'BioBART']
    colors = {'bart-cnn': '#1f77b4', 'pegasus': '#ff7f0e', 'biobart': '#2ca02c'}

    x = np.arange(len(variant_order))
    width = 0.25

    for idx, (model, label) in enumerate(zip(models, model_labels)):
        model_data = df[df['Model'] == model].set_index('Variant').loc[variant_order]
        offset = (idx - 1) * width

        bars = ax.bar(x + offset, model_data['BERTScore_F1'], width,
                     label=label, color=colors[model], alpha=0.8,
                     edgecolor='black', linewidth=1)

        # Add value labels
        for bar, val in zip(bars, model_data['BERTScore_F1']):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=8)

    # Add reference line at 1.0 (perfect similarity)
    ax.axhline(y=1.0, color='red', linestyle='--', linewidth=1.5, alpha=0.5, label='Perfect similarity')

    ax.set_xlabel('Structure Variant', fontweight='bold', fontsize=12)
    ax.set_ylabel('BERTScore F1 vs Original Output', fontweight='bold', fontsize=12)
    ax.set_title('Semantic Similarity: How Much Does Structure Manipulation Change Output?\n(Baseline: Original Structure)',
                 fontsize=13, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(variant_labels)
    ax.legend(loc='lower left', fontsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0.4, 1.05)

    # Add interpretation text
    fig.text(0.5, 0.02, 'Higher scores = variant produces more similar output to original structure',
             ha='center', fontsize=10, style='italic',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(OUTPUT_DIR / 'bertscore_deviation.png', bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'bertscore_deviation.pdf', bbox_inches='tight')
    print("✓ Saved: BERTScore deviation plot")
    plt.close()


def plot_length_changes(df):
    """
    Plot how structure manipulation affects output length.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    variant_order = ['no_headers', 'no_formatting', 'shuffled']
    variant_labels = ['No Headers', 'No Formatting', 'Shuffled']

    models = ['bart-cnn', 'pegasus', 'biobart']
    model_labels = ['BART-CNN', 'PEGASUS', 'BioBART']
    colors = {'bart-cnn': '#1f77b4', 'pegasus': '#ff7f0e', 'biobart': '#2ca02c'}

    x = np.arange(len(variant_order))
    width = 0.25

    for idx, (model, label) in enumerate(zip(models, model_labels)):
        model_data = df[df['Model'] == model].set_index('Variant').loc[variant_order]
        offset = (idx - 1) * width

        bars = ax.bar(x + offset, model_data['Length_Change_Pct'], width,
                     label=label, color=colors[model], alpha=0.8,
                     edgecolor='black', linewidth=1)

        # Add value labels
        for bar, val in zip(bars, model_data['Length_Change_Pct']):
            height = bar.get_height()
            y_pos = height + 1 if height > 0 else height - 1
            ax.text(bar.get_x() + bar.get_width()/2., y_pos,
                   f'{val:+.1f}%', ha='center', va='bottom' if height > 0 else 'top',
                   fontsize=8, fontweight='bold')

    # Add reference line at 0 (no change)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1.5, alpha=0.5)

    ax.set_xlabel('Structure Variant', fontweight='bold', fontsize=12)
    ax.set_ylabel('Length Change vs Original (%)', fontweight='bold', fontsize=12)
    ax.set_title('Output Length Change from Original Structure\n(Baseline: Original Structure)',
                 fontsize=13, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(variant_labels)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Add interpretation text
    fig.text(0.5, 0.02, 'Positive = longer summaries than original, Negative = shorter summaries than original',
             ha='center', fontsize=10, style='italic',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(OUTPUT_DIR / 'length_changes.png', bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'length_changes.pdf', bbox_inches='tight')
    print("✓ Saved: Length changes plot")
    plt.close()


def plot_term_overlap(df):
    """
    Plot medical term retention across variants.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    variant_order = ['no_headers', 'no_formatting', 'shuffled']
    variant_labels = ['No Headers', 'No Formatting', 'Shuffled']

    models = ['bart-cnn', 'pegasus', 'biobart']
    model_labels = ['BART-CNN', 'PEGASUS', 'BioBART']
    colors = {'bart-cnn': '#1f77b4', 'pegasus': '#ff7f0e', 'biobart': '#2ca02c'}

    x = np.arange(len(variant_order))
    width = 0.25

    for idx, (model, label) in enumerate(zip(models, model_labels)):
        model_data = df[df['Model'] == model].set_index('Variant').loc[variant_order]
        offset = (idx - 1) * width

        bars = ax.bar(x + offset, model_data['Medical_Term_Overlap'] * 100, width,
                     label=label, color=colors[model], alpha=0.8,
                     edgecolor='black', linewidth=1)

        # Add value labels
        for bar, val in zip(bars, model_data['Medical_Term_Overlap']):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{val:.1%}', ha='center', va='bottom', fontsize=8, fontweight='bold')

    # Add reference line at 100% (perfect retention)
    ax.axhline(y=100, color='green', linestyle='--', linewidth=1.5, alpha=0.5, label='Perfect retention')

    ax.set_xlabel('Structure Variant', fontweight='bold', fontsize=12)
    ax.set_ylabel('Medical Term Overlap with Original (%)', fontweight='bold', fontsize=12)
    ax.set_title('Medical Term Retention: Do Variants Preserve Key Clinical Concepts?\n(Baseline: Original Structure)',
                 fontsize=13, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(variant_labels)
    ax.legend(loc='lower left', fontsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, 110)

    # Add interpretation text
    fig.text(0.5, 0.02, 'Higher scores = variant uses similar medical terms as original structure output',
             ha='center', fontsize=10, style='italic',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(OUTPUT_DIR / 'term_overlap.png', bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'term_overlap.pdf', bbox_inches='tight')
    print("✓ Saved: Term overlap plot")
    plt.close()


def plot_comprehensive_heatmap(df):
    """
    Create heatmap showing all metrics at once.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    variant_order = ['no_headers', 'no_formatting', 'shuffled']
    model_order = ['bart-cnn', 'pegasus', 'biobart']

    metrics = [
        ('BERTScore_F1', 'Semantic Similarity\n(BERTScore F1)', 'RdYlGn', 0.7, 1.0),
        ('Medical_Term_Overlap', 'Medical Term Overlap', 'RdYlGn', 0.5, 1.0),
        ('Length_Change_Pct', 'Length Change (%)', 'RdBu_r', -15, 15)
    ]

    for ax, (metric, title, cmap, vmin, vmax) in zip(axes, metrics):
        pivot = df.pivot(index='Model', columns='Variant', values=metric)
        pivot = pivot.loc[model_order, variant_order]

        # Scale term overlap to percentage
        if metric == 'Medical_Term_Overlap':
            pivot = pivot * 100

        sns.heatmap(pivot, annot=True, fmt='.2f' if metric != 'Length_Change_Pct' else '.1f',
                   cmap=cmap, center=(vmin+vmax)/2, vmin=vmin, vmax=vmax,
                   linewidths=1, linecolor='black',
                   cbar_kws={'label': 'Score'},
                   ax=ax)

        ax.set_title(title, fontweight='bold', fontsize=11)
        ax.set_xlabel('Variant vs Original', fontweight='bold')
        ax.set_ylabel('Model' if ax == axes[0] else '', fontweight='bold')
        ax.set_xticklabels(['No Headers', 'No Formatting', 'Shuffled'], rotation=45, ha='right')
        ax.set_yticklabels(['BART-CNN', 'PEGASUS', 'BioBART'], rotation=0)

    fig.suptitle('Reference-Free Analysis: Structure Variants vs Original Baseline',
                 fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'comprehensive_heatmap.png', bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'comprehensive_heatmap.pdf', bbox_inches='tight')
    print("✓ Saved: Comprehensive heatmap")
    plt.close()


def main():
    print("\n" + "="*70)
    print("Reference-Free Analysis: Structure Variants vs Original Baseline")
    print("="*70)

    models = ['bart-cnn', 'pegasus', 'biobart']
    all_results = {}

    # Analyze each model
    for model in models:
        results = analyze_model(model)
        if results:
            all_results[model] = results

    if not all_results:
        print("\n✗ No results to analyze. Make sure prediction files exist in predictions/")
        print("Run: python experiment_save_predictions.py to generate prediction files")
        return

    # Create comparison dataframe
    print("\n" + "="*70)
    print("Creating comparison dataframe...")
    df = create_comparison_dataframe(all_results)

    # Save results
    df.to_csv(OUTPUT_DIR / 'reference_free_comparison.csv', index=False)
    print(f"✓ Saved: {OUTPUT_DIR / 'reference_free_comparison.csv'}")

    # Create LaTeX table
    latex_table = df.to_latex(
        index=False,
        caption='Reference-Free Comparison: Structure Variants vs Original Baseline',
        label='tab:reference_free',
        float_format='%.4f',
        column_format='llrrrrr'
    )
    with open(OUTPUT_DIR / 'reference_free_table.tex', 'w') as f:
        f.write(latex_table)
    print(f"✓ Saved: {OUTPUT_DIR / 'reference_free_table.tex'}")

    # Print summary
    print("\n" + "="*70)
    print("Summary Statistics:")
    print("="*70)
    print(df.to_string(index=False))

    # Create visualizations
    print("\n" + "="*70)
    print("Generating visualizations...")
    print("="*70 + "\n")

    plot_bertscore_deviation(df)
    plot_length_changes(df)
    plot_term_overlap(df)
    plot_comprehensive_heatmap(df)

    print("\n" + "="*70)
    print(f"✓ All reference-free analyses saved to: {OUTPUT_DIR}/")
    print("="*70)
    print("\nGenerated files:")
    print("  1. bertscore_deviation.png/pdf - Semantic similarity to original")
    print("  2. length_changes.png/pdf - Output length changes")
    print("  3. term_overlap.png/pdf - Medical term retention")
    print("  4. comprehensive_heatmap.png/pdf - All metrics combined")
    print("  5. reference_free_comparison.csv - Numerical results")
    print("  6. reference_free_table.tex - LaTeX table")
    print("\n" + "="*70)
    print("\nKey Insights:")
    print("- BERTScore F1 close to 1.0 = variant produces very similar output")
    print("- BERTScore F1 < 0.8 = significant semantic deviation")
    print("- Medical term overlap shows content preservation")
    print("- Length changes show verbosity differences")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
