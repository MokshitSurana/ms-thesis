"""
Modified Experiment Script to Save All Predictions
Run this to generate prediction files for reference-free analysis.

Usage:
    python experiment_save_predictions.py
"""

import json
import torch
from pathlib import Path
from tqdm import tqdm
import pandas as pd

from config import (
    MODELS,
    STRUCTURE_VARIANTS,
    DATA_PATH,
    SAMPLE_SIZE,
    BATCH_SIZE,
    RANDOM_SEED,
    MAX_INPUT_LENGTH,
    GENERATION_CONFIG
)
from models import load_model_and_tokenizer
from evaluation import Evaluator
from structure_manipulation import StructureManipulator

# Output directory for predictions
PREDICTIONS_DIR = Path('predictions')
PREDICTIONS_DIR.mkdir(exist_ok=True)


def save_predictions_for_variant(model_name, structure_variant, max_samples=None):
    """
    Generate and save predictions for a specific model and structure variant.
    """
    print(f"\n{'='*60}")
    print(f"Model: {model_name} | Variant: {structure_variant}")
    print('='*60)

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

    # Initialize manipulator
    manipulator = StructureManipulator()

    # Process structure variant
    print(f"Applying structure variant: {structure_variant}...")
    if structure_variant == 'original':
        inputs = df['note_text'].tolist()
    elif structure_variant == 'no_headers':
        inputs = [manipulator.remove_headers(text) for text in df['note_text']]
    elif structure_variant == 'no_formatting':
        inputs = [manipulator.remove_formatting(text) for text in df['note_text']]
    elif structure_variant == 'shuffled':
        inputs = [manipulator.shuffle_sections(text) for text in df['note_text']]

    references = df['reference_summary'].tolist()

    # Generate predictions
    print("Generating predictions...")
    predictions = []

    for i in tqdm(range(0, len(inputs), BATCH_SIZE), desc="Processing batches"):
        batch_inputs = inputs[i:i+BATCH_SIZE]
        batch_refs = references[i:i+BATCH_SIZE]

        # Tokenize
        encoded = tokenizer(
            batch_inputs,
            max_length=MAX_INPUT_LENGTH,
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
                **GENERATION_CONFIG
            )

        # Decode
        batch_preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        # Save with references
        for j, (pred, ref, inp) in enumerate(zip(batch_preds, batch_refs, batch_inputs)):
            predictions.append({
                'id': i + j,
                'input': inp[:500],  # Save first 500 chars of input for reference
                'prediction': pred,
                'reference': ref,
                'model': model_name,
                'structure_variant': structure_variant
            })

    # Save predictions to JSON
    output_file = PREDICTIONS_DIR / f'{model_name}_{structure_variant}_predictions.json'
    with open(output_file, 'w') as f:
        json.dump(predictions, f, indent=2)

    print(f"✓ Saved {len(predictions)} predictions to {output_file}")

    return len(predictions)


def main():
    """Generate and save predictions for all model-variant combinations."""
    print("\n" + "="*70)
    print("Generating Predictions for Reference-Free Analysis")
    print("="*70)

    # Select models to run
    models_to_run = ['bart-cnn', 'pegasus', 'biobart']

    total_files = 0

    for model_name in models_to_run:
        for variant in STRUCTURE_VARIANTS:
            try:
                count = save_predictions_for_variant(model_name, variant, max_samples=SAMPLE_SIZE)
                total_files += 1
                print(f"✓ Completed: {model_name} - {variant} ({count} predictions)")

            except Exception as e:
                print(f"✗ Error with {model_name} - {variant}: {str(e)}")
                continue

            # Free GPU memory
            torch.cuda.empty_cache()

    print("\n" + "="*70)
    print(f"✓ Completed! Generated {total_files} prediction files")
    print(f"✓ Saved to: {PREDICTIONS_DIR}/")
    print("="*70)
    print("\nNext steps:")
    print("  1. Run: python reference_free_analysis.py")
    print("  2. Check: figures/reference_free/ for visualizations")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
