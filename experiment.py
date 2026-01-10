"""
Main experiment runner for RQ1: Effect of Note Structure.

This script:
1. Loads MIMIC-IV clinical notes
2. Creates different structure variants (original, no_headers, no_formatting, shuffled)
3. Runs general models (BART-CNN) and medical models (Clinical-T5)
4. Evaluates using ROUGE and BERTScore
5. Saves results for analysis
"""

import pandas as pd
import torch
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm
import json
import os

from config import (
    DATA_PATH,
    MODELS,
    STRUCTURE_VARIANTS,
    OUTPUT_DIR,
    BATCH_SIZE,
    RANDOM_SEED
)
from structure_manipulation import NoteStructureManipulator
from models import ModelFactory
from evaluation import SummarizationEvaluator, format_metrics_table


class ExperimentRunner:
    """
    Run complete experiment for RQ1.
    """

    def __init__(
        self,
        data_path: str = DATA_PATH,
        output_dir: str = OUTPUT_DIR,
        n_samples: int = None
    ):
        """
        Initialize experiment.

        Args:
            data_path: Path to MIMIC dataset CSV
            output_dir: Directory for results
            n_samples: Number of samples to use (None = all)
        """
        self.data_path = data_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Load data
        print(f"Loading data from {data_path}...")
        self.df = pd.read_csv(data_path)

        if n_samples:
            self.df = self.df.head(n_samples)

        print(f"✓ Loaded {len(self.df)} clinical notes")

        # Initialize components
        self.manipulator = NoteStructureManipulator(seed=RANDOM_SEED)
        self.evaluator = SummarizationEvaluator()

        # Storage for results
        self.results = {}

    def run_single_configuration(
        self,
        model_key: str,
        structure_variant: str,
        batch_size: int = BATCH_SIZE
    ) -> Dict:
        """
        Run experiment for one model + structure variant combination.

        Args:
            model_key: Model identifier (e.g., 'bart-cnn')
            structure_variant: Structure variant (e.g., 'original')
            batch_size: Batch size for inference

        Returns:
            Dictionary with results
        """
        print(f"\n{'='*80}")
        print(f"Running: {model_key} on {structure_variant} structure")
        print(f"{'='*80}")

        # Load model
        model = ModelFactory.create_model(model_key)

        # Create structure variants
        print(f"Creating {structure_variant} variants...")
        texts = []
        for text in tqdm(self.df['text'], desc="Transforming notes"):
            variant_text = self.manipulator.apply_variant(text, structure_variant)
            texts.append(variant_text)

        # Generate summaries
        print(f"Generating summaries...")
        predictions = []

        for i in tqdm(range(0, len(texts), batch_size), desc="Inference"):
            batch = texts[i:i + batch_size]
            batch_preds = model.summarize_batch(batch)
            predictions.extend(batch_preds)

        # Get references
        references = self.df['reference_summary'].tolist()

        # Evaluate
        print(f"Evaluating...")
        metrics = self.evaluator.compute_all_metrics(
            predictions,
            references,
            include_bertscore=True  # Set False to speed up
        )

        # Add length stats
        length_stats = self.evaluator.compute_length_stats(predictions, references)
        metrics.update(length_stats)

        # Print results
        print(format_metrics_table(metrics))

        # Store results
        result = {
            'model': model_key,
            'structure_variant': structure_variant,
            'n_samples': len(texts),
            'metrics': metrics,
            'predictions': predictions[:10],  # Save first 10 for inspection
        }

        return result

    def run_full_experiment(
        self,
        models_to_test: List[str] = None,
        variants_to_test: List[str] = None
    ):
        """
        Run complete experiment across all models and variants.

        Args:
            models_to_test: List of model keys (None = all)
            variants_to_test: List of variants (None = all)
        """
        # Default to all if not specified
        if models_to_test is None:
            models_to_test = list(MODELS['general'].keys()) + list(MODELS['medical'].keys())

        if variants_to_test is None:
            variants_to_test = STRUCTURE_VARIANTS

        print(f"\n{'='*80}")
        print(f"FULL EXPERIMENT")
        print(f"{'='*80}")
        print(f"Models: {models_to_test}")
        print(f"Structure Variants: {variants_to_test}")
        print(f"Total configurations: {len(models_to_test) * len(variants_to_test)}")

        # Run all combinations
        all_results = []

        for model_key in models_to_test:
            for variant in variants_to_test:
                try:
                    result = self.run_single_configuration(model_key, variant)
                    all_results.append(result)

                    # Save intermediate results
                    self.save_results(all_results)

                except Exception as e:
                    print(f"ERROR with {model_key} + {variant}: {e}")
                    continue

        self.results = all_results
        print(f"\n✓ Experiment complete! Results saved to {self.output_dir}")

        return all_results

    def save_results(self, results: List[Dict] = None):
        """
        Save results to JSON and CSV.
        """
        if results is None:
            results = self.results

        # Save detailed JSON
        json_path = self.output_dir / 'detailed_results.json'
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)

        # Create summary DataFrame
        summary_data = []
        for r in results:
            row = {
                'model': r['model'],
                'structure_variant': r['structure_variant'],
                'n_samples': r['n_samples'],
            }
            row.update(r['metrics'])
            summary_data.append(row)

        summary_df = pd.DataFrame(summary_data)

        # Save CSV
        csv_path = self.output_dir / 'summary_results.csv'
        summary_df.to_csv(csv_path, index=False)

        print(f"✓ Results saved to {self.output_dir}")

    def create_comparison_table(self):
        """
        Create comparison table across all configurations.
        """
        if not self.results:
            print("No results to compare. Run experiment first.")
            return

        # Load summary
        summary_df = pd.read_csv(self.output_dir / 'summary_results.csv')

        # Create pivot tables for key metrics
        for metric in ['rouge1', 'rouge2', 'rougeL', 'bertscore_f1']:
            if metric in summary_df.columns:
                pivot = summary_df.pivot(
                    index='model',
                    columns='structure_variant',
                    values=metric
                )

                print(f"\n{'='*80}")
                print(f"{metric.upper()} by Model and Structure")
                print(f"{'='*80}")
                print(pivot.to_string())

                # Save to CSV
                pivot.to_csv(self.output_dir / f'comparison_{metric}.csv')


def run_quick_test():
    """
    Run quick test with 10 samples and 2 configurations.
    """
    print("Running QUICK TEST (10 samples, 2 configurations)")

    runner = ExperimentRunner(n_samples=10)

    # Test just BART-CNN with original and no_formatting
    runner.run_full_experiment(
        models_to_test=['bart-cnn'],
        variants_to_test=['original', 'no_formatting']
    )

    runner.create_comparison_table()


def run_full_experiment():
    """
    Run full experiment with all models and variants.
    """
    print("Running FULL EXPERIMENT")

    runner = ExperimentRunner()  # Use all samples

    # Run all models and variants
    runner.run_full_experiment()

    runner.create_comparison_table()


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == '--full':
        run_full_experiment()
    else:
        run_quick_test()
