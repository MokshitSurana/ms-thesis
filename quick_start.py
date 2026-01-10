"""
Quick start script - Minimal working example to test your setup.
This is similar to your original code but uses the new framework.
"""

import pandas as pd
from structure_manipulation import NoteStructureManipulator
from models import ModelFactory
from evaluation import SummarizationEvaluator


def main():
    """
    Quick test similar to your original code.
    Tests BART-CNN on 3 samples with original vs no_formatting structure.
    """

    print("\n" + "="*80)
    print("QUICK START - Testing BART-CNN on Clinical Notes")
    print("="*80)

    # Load your data (update path if needed)
    print("\nLoading data...")
    df = pd.read_csv('mimic_1000_notes_with_references.csv')
    df_test = df.head(3).copy()
    print(f"✓ Loaded {len(df_test)} notes for testing")

    # Initialize components
    print("\nInitializing components...")
    manipulator = NoteStructureManipulator()
    model = ModelFactory.create_model('bart-cnn')
    evaluator = SummarizationEvaluator()

    # Test on original vs no_formatting
    for variant_name in ['original', 'no_formatting']:

        print(f"\n{'='*80}")
        print(f"TESTING: {variant_name.upper()} STRUCTURE")
        print(f"{'='*80}")

        predictions = []
        references = []

        for idx, row in df_test.iterrows():
            # Transform note
            text_variant = manipulator.apply_variant(row['text'], variant_name)

            # Show first note input
            if idx == 0:
                print(f"\nSample Input (first 300 chars):")
                print(text_variant[:300] + "...")

            # Generate summary
            summary = model.summarize(text_variant)

            predictions.append(summary)
            references.append(row['reference_summary'])

            # Show first note output
            if idx == 0:
                print(f"\nGenerated Summary:")
                print(summary)
                print(f"\nReference Summary:")
                print(row['reference_summary'][:300] + "...")

        # Evaluate
        print(f"\nEvaluating {variant_name}...")
        metrics = evaluator.compute_rouge(predictions, references)

        print(f"\nResults for {variant_name}:")
        print(f"  ROUGE-1: {metrics['rouge1']:.4f}")
        print(f"  ROUGE-2: {metrics['rouge2']:.4f}")
        print(f"  ROUGE-L: {metrics['rougeL']:.4f}")

    print("\n" + "="*80)
    print("✓ Quick test complete!")
    print("="*80)
    print("\nNext steps:")
    print("  1. For full experiment: python experiment.py --full")
    print("  2. For visualization: python visualize.py")
    print("  3. See README.md for more options")


if __name__ == '__main__':
    main()
