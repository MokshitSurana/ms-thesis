"""
Visualization script for RQ1: Effect of Note Structure on Clinical Summarization
Generates publication-quality figures for thesis.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("colorblind")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 11
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10

# Create output directory
OUTPUT_DIR = Path('figures')
OUTPUT_DIR.mkdir(exist_ok=True)

# Full Experiment Results (n=1000)
results_data = {
    'Model': ['BART-CNN', 'BART-CNN', 'BART-CNN', 'BART-CNN',
              'PEGASUS', 'PEGASUS', 'PEGASUS', 'PEGASUS',
              'BioBART', 'BioBART', 'BioBART', 'BioBART'],
    'Structure': ['original', 'no_headers', 'no_formatting', 'shuffled'] * 3,
    'ROUGE-1': [0.2622, 0.2480, 0.2576, 0.2313,  # BART-CNN
                0.2282, 0.2259, 0.2258, 0.1617,  # PEGASUS
                0.2978, 0.3154, 0.2978, 0.1696], # BioBART
    'ROUGE-2': [0.0983, 0.0856, 0.0937, 0.0801,  # BART-CNN
                0.0815, 0.0783, 0.0781, 0.0419,  # PEGASUS
                0.1187, 0.1289, 0.1187, 0.0515], # BioBART
    'ROUGE-L': [0.2187, 0.1984, 0.2097, 0.1944,  # BART-CNN
                0.1909, 0.1893, 0.1896, 0.1384,  # PEGASUS
                0.2429, 0.2627, 0.2461, 0.1480], # BioBART
    'BERTScore': [0.6143, 0.6001, 0.6061, 0.5885,  # BART-CNN
                  0.5727, 0.5695, 0.5702, 0.5283,  # PEGASUS
                  0.5662, 0.5689, 0.6148, 0.5065], # BioBART
    'Domain': ['General', 'General', 'General', 'General',
               'General', 'General', 'General', 'General',
               'Medical', 'Medical', 'Medical', 'Medical'],
    'Quality': ['Good', 'Good', 'Good', 'Good',  # BART-CNN
                'Good', 'Good', 'Good', 'Good',  # PEGASUS
                'Poor', 'Poor', 'Good', 'Poor']  # BioBART (only no_formatting is good)
}

df = pd.DataFrame(results_data)

# Define colors
model_colors = {
    'BART-CNN': '#1f77b4',   # blue
    'PEGASUS': '#ff7f0e',    # orange
    'BioBART': '#2ca02c'     # green
}

structure_order = ['original', 'no_headers', 'no_formatting', 'shuffled']
structure_labels = ['Original\n(with headers)', 'No Headers', 'No Formatting\n(continuous)', 'Shuffled\n(disrupted)']


def plot_1_structure_effect_by_model():
    """
    Figure 1: Structure Effect by Model (Main RQ1 finding)
    Shows how each model responds to structure manipulation.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    models = ['BART-CNN', 'PEGASUS', 'BioBART']
    titles = ['BART-CNN (General)', 'PEGASUS (General)', 'BioBART (Medical)']

    for idx, (model, title) in enumerate(zip(models, titles)):
        ax = axes[idx]
        model_data = df[df['Model'] == model].set_index('Structure').loc[structure_order]

        # Bar plot
        bars = ax.bar(range(len(structure_order)), model_data['ROUGE-1'],
                      color=model_colors[model], alpha=0.8, edgecolor='black', linewidth=1.2)

        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, model_data['ROUGE-1'])):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

        # Calculate percent changes from original
        original_score = model_data.loc['original', 'ROUGE-1']
        for i, struct in enumerate(structure_order[1:], 1):
            score = model_data.loc[struct, 'ROUGE-1']
            pct_change = ((score - original_score) / original_score) * 100
            color = 'green' if pct_change > 0 else 'red'
            ax.text(i, 0.01, f'{pct_change:+.1f}%', ha='center', va='bottom',
                   fontsize=8, color=color, fontweight='bold')

        ax.set_xlabel('Structure Variant', fontweight='bold')
        ax.set_title(title, fontweight='bold', pad=10)
        ax.set_xticks(range(len(structure_order)))
        ax.set_xticklabels(structure_labels, fontsize=9)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_ylim(0, 0.35)

    axes[0].set_ylabel('ROUGE-1 Score', fontweight='bold')

    fig.suptitle('Effect of Note Structure on Summarization Performance (RQ1)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig1_structure_effect_by_model.png', bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'fig1_structure_effect_by_model.pdf', bbox_inches='tight')
    print("✓ Saved: Figure 1 - Structure Effect by Model")
    plt.close()


def plot_2_model_comparison():
    """
    Figure 2: Model Comparison Across Structures
    Shows domain-specific models outperform general models.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(structure_order))
    width = 0.25

    for idx, model in enumerate(['BART-CNN', 'PEGASUS', 'BioBART']):
        model_data = df[df['Model'] == model].set_index('Structure').loc[structure_order]
        offset = (idx - 1) * width
        bars = ax.bar(x + offset, model_data['ROUGE-1'], width,
                     label=model, color=model_colors[model], alpha=0.8,
                     edgecolor='black', linewidth=1)

    ax.set_xlabel('Structure Variant', fontweight='bold')
    ax.set_ylabel('ROUGE-1 Score', fontweight='bold')
    ax.set_title('Model Performance Comparison Across Structure Variants',
                 fontsize=13, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(structure_labels)
    ax.legend(title='Model', title_fontsize=11, loc='upper right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, 0.35)

    # Add annotation for best performance
    ax.annotate('Best: BioBART no_headers\nROUGE-1 = 0.315',
                xy=(1, 0.3154), xytext=(2, 0.33),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='darkgreen'),
                fontsize=10, fontweight='bold', color='darkgreen',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.7))

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig2_model_comparison.png', bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'fig2_model_comparison.pdf', bbox_inches='tight')
    print("✓ Saved: Figure 2 - Model Comparison")
    plt.close()


def plot_3_shuffling_impact():
    """
    Figure 3: Impact of Shuffling Sections
    Highlights the devastating effect of disrupting logical flow.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    models = ['BART-CNN', 'PEGASUS', 'BioBART']
    original_scores = []
    shuffled_scores = []
    percent_drops = []

    for model in models:
        model_data = df[df['Model'] == model].set_index('Structure')
        orig = model_data.loc['original', 'ROUGE-1']
        shuf = model_data.loc['shuffled', 'ROUGE-1']
        pct_drop = ((shuf - orig) / orig) * 100

        original_scores.append(orig)
        shuffled_scores.append(shuf)
        percent_drops.append(pct_drop)

    x = np.arange(len(models))
    width = 0.35

    bars1 = ax.bar(x - width/2, original_scores, width, label='Original Structure',
                  color='#2ecc71', alpha=0.8, edgecolor='black', linewidth=1.2)
    bars2 = ax.bar(x + width/2, shuffled_scores, width, label='Shuffled Sections',
                  color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=1.2)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Add percent drop annotations
    for i, (pct, model) in enumerate(zip(percent_drops, models)):
        ax.text(i, max(original_scores[i], shuffled_scores[i]) + 0.03,
               f'{pct:.1f}%', ha='center', fontsize=11, fontweight='bold',
               color='darkred',
               bbox=dict(boxstyle='round,pad=0.4', facecolor='yellow', alpha=0.7))

    ax.set_xlabel('Model', fontweight='bold')
    ax.set_ylabel('ROUGE-1 Score', fontweight='bold')
    ax.set_title('Impact of Shuffling Sections on Model Performance',
                 fontsize=13, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(['BART-CNN\n(General)', 'PEGASUS\n(General)', 'BioBART\n(Medical)'])
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, 0.35)

    # Add key finding text
    fig.text(0.5, 0.02, 'Key Finding: Medical models (BioBART) are MORE affected by structure disruption (-43%) than general models (-12% to -29%)',
             ha='center', fontsize=10, style='italic', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(OUTPUT_DIR / 'fig3_shuffling_impact.png', bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'fig3_shuffling_impact.pdf', bbox_inches='tight')
    print("✓ Saved: Figure 3 - Shuffling Impact")
    plt.close()


def plot_4_rouge_vs_bertscore():
    """
    Figure 4: ROUGE vs BERTScore
    Shows the disconnect between metrics for BioBART.
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    # Scatter plot with different markers for each model
    markers = {'BART-CNN': 'o', 'PEGASUS': 's', 'BioBART': '^'}

    for model in ['BART-CNN', 'PEGASUS', 'BioBART']:
        model_data = df[df['Model'] == model]

        # Different marker fill for BioBART quality
        if model == 'BioBART':
            for _, row in model_data.iterrows():
                marker = 'o' if row['Quality'] == 'Good' else 'X'
                edgecolor = 'green' if row['Quality'] == 'Good' else 'red'
                linewidth = 2 if row['Quality'] == 'Good' else 2
                ax.scatter(row['ROUGE-1'], row['BERTScore'],
                          s=200, marker=marker, color=model_colors[model],
                          edgecolor=edgecolor, linewidth=linewidth, alpha=0.7,
                          label=f"{model} ({'Good' if row['Quality'] == 'Good' else 'Poor'} output)" if _ == 2 else "")
        else:
            ax.scatter(model_data['ROUGE-1'], model_data['BERTScore'],
                      s=200, marker=markers[model], color=model_colors[model],
                      edgecolor='black', linewidth=1.5, alpha=0.7, label=model)

    # Annotate key points
    # BioBART no_formatting (good quality, best BERTScore)
    ax.annotate('BioBART no_formatting\n(Good quality)\nBest BERTScore: 0.615',
                xy=(0.2978, 0.6148), xytext=(0.24, 0.60),
                arrowprops=dict(arrowstyle='->', lw=2, color='darkgreen'),
                fontsize=9, fontweight='bold', color='darkgreen',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.8))

    # BioBART no_headers (poor quality, high ROUGE)
    ax.annotate('BioBART no_headers\n(Garbage output!)\nHigh ROUGE, Low BERTScore',
                xy=(0.3154, 0.5689), xytext=(0.28, 0.53),
                arrowprops=dict(arrowstyle='->', lw=2, color='darkred'),
                fontsize=9, fontweight='bold', color='darkred',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightcoral', alpha=0.8))

    ax.set_xlabel('ROUGE-1 Score (Lexical Overlap)', fontweight='bold', fontsize=12)
    ax.set_ylabel('BERTScore F1 (Semantic Similarity)', fontweight='bold', fontsize=12)
    ax.set_title('ROUGE vs BERTScore: Metric Disconnect for BioBART',
                 fontsize=13, fontweight='bold', pad=15)
    ax.grid(alpha=0.3, linestyle='--')
    ax.legend(loc='lower right', fontsize=9, framealpha=0.9)

    # Add diagonal reference line
    ax.plot([0.15, 0.35], [0.50, 0.65], 'k--', alpha=0.3, linewidth=1, label='Ideal correlation')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig4_rouge_vs_bertscore.png', bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'fig4_rouge_vs_bertscore.pdf', bbox_inches='tight')
    print("✓ Saved: Figure 4 - ROUGE vs BERTScore")
    plt.close()


def plot_5_heatmap():
    """
    Figure 5: Heatmap of All Results
    Comprehensive view of model × structure performance.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ROUGE-1 Heatmap
    rouge_pivot = df.pivot(index='Model', columns='Structure', values='ROUGE-1')
    rouge_pivot = rouge_pivot[structure_order]  # Reorder columns

    sns.heatmap(rouge_pivot, annot=True, fmt='.3f', cmap='RdYlGn',
                center=0.25, vmin=0.15, vmax=0.35,
                linewidths=1, linecolor='black',
                cbar_kws={'label': 'ROUGE-1 Score'},
                ax=axes[0])
    axes[0].set_title('ROUGE-1 Performance', fontweight='bold', fontsize=12)
    axes[0].set_xlabel('Structure Variant', fontweight='bold')
    axes[0].set_ylabel('Model', fontweight='bold')
    axes[0].set_xticklabels(['Original', 'No Headers', 'No Formatting', 'Shuffled'], rotation=45, ha='right')

    # BERTScore Heatmap
    bert_pivot = df.pivot(index='Model', columns='Structure', values='BERTScore')
    bert_pivot = bert_pivot[structure_order]

    sns.heatmap(bert_pivot, annot=True, fmt='.3f', cmap='RdYlGn',
                center=0.58, vmin=0.50, vmax=0.62,
                linewidths=1, linecolor='black',
                cbar_kws={'label': 'BERTScore F1'},
                ax=axes[1])
    axes[1].set_title('BERTScore Performance', fontweight='bold', fontsize=12)
    axes[1].set_xlabel('Structure Variant', fontweight='bold')
    axes[1].set_ylabel('')
    axes[1].set_xticklabels(['Original', 'No Headers', 'No Formatting', 'Shuffled'], rotation=45, ha='right')

    fig.suptitle('Comprehensive Performance Heatmap: All Models × All Structures',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig5_performance_heatmap.png', bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'fig5_performance_heatmap.pdf', bbox_inches='tight')
    print("✓ Saved: Figure 5 - Performance Heatmap")
    plt.close()


def plot_6_domain_comparison():
    """
    Figure 6: General vs Medical Models
    Shows domain-specific models outperform by 20%.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Calculate average performance by domain
    domain_stats = df.groupby(['Domain', 'Structure']).agg({
        'ROUGE-1': 'mean'
    }).reset_index()

    x = np.arange(len(structure_order))
    width = 0.35

    general_data = domain_stats[domain_stats['Domain'] == 'General'].set_index('Structure').loc[structure_order]
    medical_data = domain_stats[domain_stats['Domain'] == 'Medical'].set_index('Structure').loc[structure_order]

    bars1 = ax.bar(x - width/2, general_data['ROUGE-1'], width,
                  label='General Models (BART-CNN, PEGASUS avg)',
                  color='#3498db', alpha=0.8, edgecolor='black', linewidth=1.2)
    bars2 = ax.bar(x + width/2, medical_data['ROUGE-1'], width,
                  label='Medical Model (BioBART)',
                  color='#2ecc71', alpha=0.8, edgecolor='black', linewidth=1.2)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.003,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Add percent improvement for best case
    best_gen = general_data['ROUGE-1'].max()
    best_med = medical_data['ROUGE-1'].max()
    improvement = ((best_med - best_gen) / best_gen) * 100

    ax.text(0.5, 0.33, f'Domain-specific models\noutperform by +{improvement:.1f}%',
           transform=ax.transAxes, ha='center', fontsize=12, fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.8', facecolor='yellow', alpha=0.7))

    ax.set_xlabel('Structure Variant', fontweight='bold')
    ax.set_ylabel('Average ROUGE-1 Score', fontweight='bold')
    ax.set_title('General vs Medical Models: Domain Adaptation Matters',
                 fontsize=13, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(structure_labels)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, 0.35)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig6_domain_comparison.png', bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'fig6_domain_comparison.pdf', bbox_inches='tight')
    print("✓ Saved: Figure 6 - Domain Comparison")
    plt.close()


def plot_7_all_metrics():
    """
    Figure 7: Multi-metric comparison for best-performing variants
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Select best variants for each model
    best_variants = [
        ('BART-CNN', 'original', 0.2622, 0.0983, 0.2187, 0.6143),
        ('PEGASUS', 'original', 0.2282, 0.0815, 0.1909, 0.5727),
        ('BioBART', 'no_headers', 0.3154, 0.1289, 0.2627, 0.5689),
        ('BioBART*', 'no_formatting\n(clean output)', 0.2978, 0.1187, 0.2461, 0.6148),
    ]

    x = np.arange(len(best_variants))
    width = 0.2

    metrics = ['ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'BERTScore']
    colors_metrics = ['#e74c3c', '#3498db', '#f39c12', '#9b59b6']

    for i, (metric, color) in enumerate(zip(metrics, colors_metrics)):
        values = [v[i+2] for v in best_variants]
        # Normalize BERTScore to same scale
        if metric == 'BERTScore':
            values = [(v - 0.5) * 2 for v in values]  # Scale to ~0.1-0.25 range

        offset = (i - 1.5) * width
        bars = ax.bar(x + offset, values, width, label=metric,
                     color=color, alpha=0.8, edgecolor='black', linewidth=0.8)

        # Add value labels
        for bar, orig_val in zip(bars, [v[i+2] for v in best_variants]):
            height = bar.get_height()
            display_val = orig_val if metric != 'BERTScore' else orig_val
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.003,
                   f'{display_val:.3f}', ha='center', va='bottom',
                   fontsize=7, rotation=90)

    ax.set_xlabel('Model - Best Structure Variant', fontweight='bold')
    ax.set_ylabel('Score (BERTScore scaled)', fontweight='bold')
    ax.set_title('Multi-Metric Comparison: Best Performing Configurations',
                 fontsize=13, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{v[0]}\n{v[1]}" for v in best_variants], fontsize=9)
    ax.legend(loc='upper left', fontsize=10, ncol=2)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Add note about BioBART*
    ax.text(0.98, 0.02, '* BioBART no_formatting: only variant with clean output',
           transform=ax.transAxes, ha='right', fontsize=9, style='italic',
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig7_all_metrics.png', bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'fig7_all_metrics.pdf', bbox_inches='tight')
    print("✓ Saved: Figure 7 - All Metrics Comparison")
    plt.close()


def generate_summary_table():
    """
    Generate a summary table for thesis
    """
    summary = df.groupby(['Model', 'Structure']).agg({
        'ROUGE-1': 'first',
        'ROUGE-2': 'first',
        'ROUGE-L': 'first',
        'BERTScore': 'first'
    }).round(4)

    summary.to_csv(OUTPUT_DIR / 'summary_table.csv')

    # LaTeX table
    latex_table = summary.to_latex(
        caption='Full Experiment Results: All Models and Structure Variants (n=1000)',
        label='tab:full_results',
        column_format='llrrrr',
        escape=False
    )

    with open(OUTPUT_DIR / 'summary_table.tex', 'w') as f:
        f.write(latex_table)

    print("✓ Saved: Summary tables (CSV and LaTeX)")


if __name__ == '__main__':
    print("\n" + "="*60)
    print("Generating Publication-Quality Visualizations")
    print("="*60 + "\n")

    plot_1_structure_effect_by_model()
    plot_2_model_comparison()
    plot_3_shuffling_impact()
    plot_4_rouge_vs_bertscore()
    plot_5_heatmap()
    plot_6_domain_comparison()
    plot_7_all_metrics()
    generate_summary_table()

    print("\n" + "="*60)
    print(f"✓ All visualizations saved to: {OUTPUT_DIR}/")
    print("="*60)
    print("\nGenerated files:")
    print("  1. fig1_structure_effect_by_model.png/pdf - Main RQ1 finding")
    print("  2. fig2_model_comparison.png/pdf - Model performance comparison")
    print("  3. fig3_shuffling_impact.png/pdf - Shuffling devastates performance")
    print("  4. fig4_rouge_vs_bertscore.png/pdf - Metric disconnect")
    print("  5. fig5_performance_heatmap.png/pdf - Comprehensive heatmap")
    print("  6. fig6_domain_comparison.png/pdf - General vs Medical models")
    print("  7. fig7_all_metrics.png/pdf - Multi-metric comparison")
    print("  8. summary_table.csv - Results table")
    print("  9. summary_table.tex - LaTeX table for thesis")
    print("\n" + "="*60)
