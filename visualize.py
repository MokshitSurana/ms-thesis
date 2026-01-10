"""
Visualization and analysis utilities for RQ1 results.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

# Set style
sns.set_theme(style='whitegrid', context='paper')
plt.rcParams['figure.figsize'] = (12, 6)


class ResultsVisualizer:
    """
    Create visualizations from experiment results.
    """

    def __init__(self, results_dir: str = 'results'):
        """
        Initialize with results directory.
        """
        self.results_dir = Path(results_dir)
        self.summary_df = pd.read_csv(self.results_dir / 'summary_results.csv')

        # Create figures directory
        self.figures_dir = self.results_dir / 'figures'
        self.figures_dir.mkdir(exist_ok=True)

    def plot_metric_comparison(
        self,
        metric: str = 'rouge1',
        save: bool = True
    ):
        """
        Create bar plot comparing metric across models and structures.
        """
        fig, ax = plt.subplots(figsize=(14, 6))

        # Prepare data
        pivot_data = self.summary_df.pivot(
            index='structure_variant',
            columns='model',
            values=metric
        )

        # Plot grouped bar chart
        pivot_data.plot(kind='bar', ax=ax, width=0.8)

        ax.set_title(f'{metric.upper()} Comparison: Models vs. Note Structure', fontsize=14, fontweight='bold')
        ax.set_xlabel('Structure Variant', fontsize=12)
        ax.set_ylabel(metric.upper(), fontsize=12)
        ax.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(axis='y', alpha=0.3)

        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        if save:
            plt.savefig(self.figures_dir / f'{metric}_comparison.png', dpi=300, bbox_inches='tight')
            print(f"✓ Saved {metric}_comparison.png")

        plt.show()

    def plot_all_rouge_metrics(self, save: bool = True):
        """
        Create faceted plot for all ROUGE metrics.
        """
        metrics = ['rouge1', 'rouge2', 'rougeL']

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        for idx, metric in enumerate(metrics):
            ax = axes[idx]

            pivot_data = self.summary_df.pivot(
                index='structure_variant',
                columns='model',
                values=metric
            )

            pivot_data.plot(kind='bar', ax=ax, width=0.8, legend=(idx == 2))

            ax.set_title(f'{metric.upper()}', fontsize=12, fontweight='bold')
            ax.set_xlabel('Structure Variant', fontsize=10)
            ax.set_ylabel('Score', fontsize=10)
            ax.grid(axis='y', alpha=0.3)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

        # Add single legend
        handles, labels = axes[-1].get_legend_handles_labels()
        fig.legend(handles, labels, loc='center right', bbox_to_anchor=(1.15, 0.5), title='Model')

        for ax in axes:
            ax.get_legend().remove()

        plt.suptitle('ROUGE Metrics Comparison Across Models and Structures', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()

        if save:
            plt.savefig(self.figures_dir / 'all_rouge_metrics.png', dpi=300, bbox_inches='tight')
            print(f"✓ Saved all_rouge_metrics.png")

        plt.show()

    def plot_heatmap(self, metric: str = 'rouge1', save: bool = True):
        """
        Create heatmap for metric across models and structures.
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        pivot_data = self.summary_df.pivot(
            index='model',
            columns='structure_variant',
            values=metric
        )

        sns.heatmap(
            pivot_data,
            annot=True,
            fmt='.3f',
            cmap='RdYlGn',
            center=pivot_data.mean().mean(),
            ax=ax,
            cbar_kws={'label': metric.upper()}
        )

        ax.set_title(f'{metric.upper()} Heatmap: Impact of Structure on Model Performance', fontsize=14, fontweight='bold')
        ax.set_xlabel('Structure Variant', fontsize=12)
        ax.set_ylabel('Model', fontsize=12)

        plt.tight_layout()

        if save:
            plt.savefig(self.figures_dir / f'{metric}_heatmap.png', dpi=300, bbox_inches='tight')
            print(f"✓ Saved {metric}_heatmap.png")

        plt.show()

    def plot_structure_effect(self, save: bool = True):
        """
        Plot effect of structure degradation on performance.
        Shows how removing structure impacts scores.
        """
        # Define structure order from most to least structured
        structure_order = ['original', 'no_headers', 'no_formatting', 'shuffled']

        # Filter to this order
        ordered_df = self.summary_df[
            self.summary_df['structure_variant'].isin(structure_order)
        ].copy()

        fig, ax = plt.subplots(figsize=(12, 6))

        for model in ordered_df['model'].unique():
            model_data = ordered_df[ordered_df['model'] == model]

            # Order by structure
            model_data = model_data.set_index('structure_variant').reindex(structure_order)

            ax.plot(
                structure_order,
                model_data['rouge1'],
                marker='o',
                linewidth=2,
                markersize=8,
                label=model
            )

        ax.set_xlabel('Structure Variant (Most → Least Structured)', fontsize=12)
        ax.set_ylabel('ROUGE-1 F1 Score', fontsize=12)
        ax.set_title('Effect of Structure Degradation on Summarization Performance', fontsize=14, fontweight='bold')
        ax.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)

        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        if save:
            plt.savefig(self.figures_dir / 'structure_degradation_effect.png', dpi=300, bbox_inches='tight')
            print(f"✓ Saved structure_degradation_effect.png")

        plt.show()

    def generate_summary_stats(self):
        """
        Generate statistical summary of results.
        """
        print("\n" + "="*80)
        print("SUMMARY STATISTICS")
        print("="*80)

        # Overall performance by model
        print("\nAverage ROUGE-1 by Model (across all structures):")
        print("-" * 80)
        model_avg = self.summary_df.groupby('model')['rouge1'].agg(['mean', 'std'])
        print(model_avg.to_string())

        # Overall performance by structure
        print("\n\nAverage ROUGE-1 by Structure (across all models):")
        print("-" * 80)
        structure_avg = self.summary_df.groupby('structure_variant')['rouge1'].agg(['mean', 'std'])
        print(structure_avg.to_string())

        # Best performing combinations
        print("\n\nTop 5 Performing Combinations:")
        print("-" * 80)
        top_5 = self.summary_df.nlargest(5, 'rouge1')[['model', 'structure_variant', 'rouge1', 'rouge2', 'rougeL']]
        print(top_5.to_string(index=False))

        # Structure impact analysis
        print("\n\nStructure Impact Analysis:")
        print("-" * 80)

        for model in self.summary_df['model'].unique():
            model_data = self.summary_df[self.summary_df['model'] == model]

            original_score = model_data[model_data['structure_variant'] == 'original']['rouge1'].values
            worst_score = model_data['rouge1'].min()

            if len(original_score) > 0:
                drop_pct = ((original_score[0] - worst_score) / original_score[0]) * 100
                print(f"{model:20s}: {drop_pct:5.2f}% performance drop (original → worst structure)")

        print("="*80)

    def generate_all_visualizations(self):
        """
        Generate all visualizations at once.
        """
        print("Generating all visualizations...")

        self.plot_all_rouge_metrics()
        self.plot_heatmap('rouge1')
        self.plot_heatmap('bertscore_f1')
        self.plot_structure_effect()
        self.generate_summary_stats()

        print(f"\n✓ All visualizations saved to {self.figures_dir}")


def main():
    """
    Generate visualizations from experiment results.
    """
    visualizer = ResultsVisualizer()
    visualizer.generate_all_visualizations()


if __name__ == '__main__':
    main()
