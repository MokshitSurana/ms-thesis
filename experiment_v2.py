"""
Updated Experiment Script for Type 1 vs Type 2 Model Comparison

Handles:
- Task-specific models (BART-CNN, PEGASUS) - no prompts
- Instruction-tuned models (Flan-T5, SciFive) - with prompts
"""

import torch
import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

from config_v2 import (
    ACTIVE_MODELS,
    STRUCTURE_VARIANTS,
    DATA_PATH,
    SAMPLE_SIZE,
    BATCH_SIZE,
    RANDOM_SEED,
    OUTPUT_DIR,
    PREDICTIONS_DIR,
    get_model_prompt,
    get_model_max_length,
    get_model_generation_config,
    get_model_type,
)

from models import load_model_and_tokenizer
from evaluation import Evaluator
from structure_manipulation import StructureManipulator


class ExperimentRunnerV2:
    """
    Enhanced experiment runner for Type 1 vs Type 2 model comparison.
    """

    def __init__(self, output_dir: str = OUTPUT_DIR):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.predictions_dir = Path(PREDICTIONS_DIR)
        self.predictions_dir.mkdir(exist_ok=True)

        self.evaluator = Evaluator()
        self.manipulator = StructureManipulator()

        print(f"\n{'='*70}")
        print("Experiment Runner V2: Type 1 vs Type 2 Model Comparison")
        print('='*70)
        print(f"Output directory: {self.output_dir}")
        print(f"Predictions directory: {self.predictions_dir}")

    def prepare_inputs(self, texts, model_name):
        """
        Prepare inputs for model, adding prompts if instruction-tuned.

        Args:
            texts: List of clinical notes
            model_name: Name of model (e.g., 'flan-t5-xl')

        Returns:
            List of prepared inputs (with prompts if needed)
        """
        # Check if model needs prompt
        model_type = get_model_type(model_name)

        if 'Instruction-Tuned' in model_type:
            # Add instruction prompt
            return [get_model_prompt(model_name, text) for text in texts]
        else:
            # No prompt needed for task-specific models
            return texts

    def run_model_on_variant(self, model_name, structure_variant, max_samples=None):
        """
        Run a single model on a single structure variant.
        """
        print(f"\n{'='*70}")
        print(f"Model: {model_name} | Variant: {structure_variant}")
        print(f"Type: {get_model_type(model_name)}")
        print('='*70)

        # Load data
        print("Loading data...")
        df = pd.read_csv(DATA_PATH)
        if max_samples:
            df = df.head(max_samples)
        print(f"✓ Loaded {len(df)} examples")

        # Load model
        print(f"Loading model: {model_name}...")
        model, tokenizer = load_model_and_tokenizer(model_name)
        model.eval()
        print(f"✓ Model loaded")

        # Get model-specific settings
        max_input_length = get_model_max_length(model_name)
        generation_config = get_model_generation_config(model_name)

        print(f"Max input length: {max_input_length}")
        print(f"Generation config: {generation_config}")

        # Apply structure manipulation
        print(f"Applying structure variant: {structure_variant}...")
        if structure_variant == 'original':
            inputs = df['note_text'].tolist()
        elif structure_variant == 'no_headers':
            inputs = [self.manipulator.remove_headers(text) for text in df['note_text']]
        elif structure_variant == 'no_formatting':
            inputs = [self.manipulator.remove_formatting(text) for text in df['note_text']]
        elif structure_variant == 'shuffled':
            inputs = [self.manipulator.shuffle_sections(text) for text in df['note_text']]

        # Prepare inputs (add prompts if instruction-tuned)
        inputs = self.prepare_inputs(inputs, model_name)

        references = df['reference_summary'].tolist()

        # Generate predictions
        print("Generating predictions...")
        predictions = []
        all_predictions_data = []  # For saving to JSON

        for i in tqdm(range(0, len(inputs), BATCH_SIZE), desc="Processing batches"):
            batch_inputs = inputs[i:i+BATCH_SIZE]
            batch_refs = references[i:i+BATCH_SIZE]

            # Tokenize
            encoded = tokenizer(
                batch_inputs,
                max_length=max_input_length,
                padding=True,
                truncation=True,
                return_tensors='pt'
            )

            # Move to GPU if available
            device = next(model.parameters()).device
            encoded = {k: v.to(device) for k, v in encoded.items()}

            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    **encoded,
                    **generation_config
                )

            # Decode
            batch_preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            predictions.extend(batch_preds)

            # Save for reference-free analysis
            for j, (pred, ref) in enumerate(zip(batch_preds, batch_refs)):
                all_predictions_data.append({
                    'id': i + j,
                    'input': batch_inputs[j][:500],  # First 500 chars
                    'prediction': pred,
                    'reference': ref,
                    'model': model_name,
                    'model_type': get_model_type(model_name),
                    'structure_variant': structure_variant
                })

        # Save all predictions for reference-free analysis
        pred_file = self.predictions_dir / f'{model_name}_{structure_variant}_predictions.json'
        with open(pred_file, 'w') as f:
            json.dump(all_predictions_data, f, indent=2)
        print(f"✓ Saved predictions to {pred_file}")

        # Evaluate
        print("Evaluating...")
        metrics = self.evaluator.compute_rouge(predictions, references)
        bertscore = self.evaluator.compute_bertscore(predictions, references)
        metrics.update(bertscore)

        length_stats = self.evaluator.compute_length_stats(predictions, references)

        # Compile results
        result = {
            'model': model_name,
            'model_type': get_model_type(model_name),
            'structure_variant': structure_variant,
            'n_samples': len(predictions),
            **metrics,
            **length_stats,
            'sample_predictions': predictions[:10],  # First 10 for inspection
        }

        # Cleanup
        del model
        torch.cuda.empty_cache()

        return result

    def run_full_experiment(self):
        """
        Run all models on all structure variants.
        """
        print("\n" + "="*70)
        print("Starting Full Experiment")
        print("="*70)
        print(f"Models: {list(ACTIVE_MODELS.keys())}")
        print(f"Variants: {STRUCTURE_VARIANTS}")
        print(f"Total experiments: {len(ACTIVE_MODELS) * len(STRUCTURE_VARIANTS)}")
        print("="*70 + "\n")

        all_results = []
        start_time = datetime.now()

        for model_name in ACTIVE_MODELS.keys():
            for variant in STRUCTURE_VARIANTS:
                try:
                    result = self.run_model_on_variant(
                        model_name,
                        variant,
                        max_samples=SAMPLE_SIZE
                    )
                    all_results.append(result)

                    # Save intermediate results
                    self.save_results(all_results)

                except Exception as e:
                    print(f"\n✗ Error with {model_name} - {variant}: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    continue

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds() / 60

        print(f"\n{'='*70}")
        print(f"✓ Experiment complete! Duration: {duration:.1f} minutes")
        print(f"✓ Results saved to {self.output_dir}")
        print("="*70)

        return all_results

    def save_results(self, results):
        """Save results to JSON and CSV."""
        if not results:
            return

        # Save detailed JSON
        json_path = self.output_dir / 'detailed_results_v2.json'
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)

        # Create summary DataFrame
        summary_data = []
        for r in results:
            summary_data.append({
                'model': r['model'],
                'model_type': r['model_type'],
                'structure_variant': r['structure_variant'],
                'rouge1': r['rouge1'],
                'rouge2': r['rouge2'],
                'rougeL': r['rougeL'],
                'bertscore_f1': r['bertscore_f1'],
                'pred_length_mean': r['pred_length_mean'],
                'ref_length_mean': r['ref_length_mean'],
            })

        df = pd.DataFrame(summary_data)

        # Save CSV
        csv_path = self.output_dir / 'summary_results_v2.csv'
        df.to_csv(csv_path, index=False)

        print(f"✓ Results saved to {self.output_dir}")

    def analyze_results(self):
        """
        Analyze results and create comparison tables.
        """
        summary_df = pd.read_csv(self.output_dir / 'summary_results_v2.csv')

        print("\n" + "="*70)
        print("Results Summary")
        print("="*70 + "\n")

        # Group by model type
        print("Performance by Model Type:")
        print("-"*70)
        type_summary = summary_df.groupby('model_type')[['rouge1', 'bertscore_f1']].mean()
        print(type_summary)

        print("\n\nRobustness Analysis (Shuffled vs Original):")
        print("-"*70)

        for model_type in summary_df['model_type'].unique():
            type_data = summary_df[summary_df['model_type'] == model_type]

            for model_name in type_data['model'].unique():
                model_data = type_data[type_data['model'] == model_name]

                orig = model_data[model_data['structure_variant'] == 'original']['rouge1'].values
                shuf = model_data[model_data['structure_variant'] == 'shuffled']['rouge1'].values

                if len(orig) > 0 and len(shuf) > 0:
                    drop = ((shuf[0] - orig[0]) / orig[0]) * 100
                    print(f"  {model_name:15s}: {orig[0]:.4f} → {shuf[0]:.4f} ({drop:+.1f}%)")

        # Create pivot tables
        for metric in ['rouge1', 'bertscore_f1']:
            pivot = summary_df.pivot(
                index='model',
                columns='structure_variant',
                values=metric
            )

            # Reorder columns
            pivot = pivot[STRUCTURE_VARIANTS]

            print(f"\n\n{metric.upper()} Scores:")
            print("-"*70)
            print(pivot.to_string())

            # Save to CSV
            pivot.to_csv(self.output_dir / f'comparison_{metric}_v2.csv')

        print("\n" + "="*70 + "\n")


def main():
    """Run the updated experiment."""
    torch.manual_seed(RANDOM_SEED)

    runner = ExperimentRunnerV2()

    # Run full experiment
    results = runner.run_full_experiment()

    # Analyze results
    runner.analyze_results()


if __name__ == '__main__':
    main()
