"""
Test script to verify setup and demonstrate functionality.
Run this before starting the full experiment.
"""

import sys


def test_imports():
    """Test all required imports."""
    print("Testing imports...")

    try:
        import pandas as pd
        print("  ✓ pandas")
    except ImportError:
        print("  ✗ pandas - run: pip install pandas")
        return False

    try:
        import torch
        print(f"  ✓ torch (version {torch.__version__})")
        print(f"    CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"    GPU: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("  ✗ torch - run: pip install torch")
        return False

    try:
        import transformers
        print(f"  ✓ transformers (version {transformers.__version__})")
    except ImportError:
        print("  ✗ transformers - run: pip install transformers")
        return False

    try:
        from rouge_score import rouge_scorer
        print("  ✓ rouge_score")
    except ImportError:
        print("  ✗ rouge_score - run: pip install rouge-score")
        return False

    try:
        import bert_score
        print("  ✓ bert_score")
    except ImportError:
        print("  ✗ bert_score - run: pip install bert-score")
        return False

    try:
        import matplotlib
        print("  ✓ matplotlib")
    except ImportError:
        print("  ✗ matplotlib - run: pip install matplotlib")
        return False

    try:
        import seaborn
        print("  ✓ seaborn")
    except ImportError:
        print("  ✗ seaborn - run: pip install seaborn")
        return False

    print("\n✓ All imports successful!\n")
    return True


def test_modules():
    """Test custom modules."""
    print("Testing custom modules...")

    try:
        from structure_manipulation import NoteStructureManipulator
        print("  ✓ structure_manipulation")

        # Quick test
        manipulator = NoteStructureManipulator()
        test_text = "Chief Complaint: Test\n\nHistory: Some history here."

        variants = manipulator.create_all_variants(test_text)
        assert len(variants) == 4
        print("    ✓ Structure variants working")

    except Exception as e:
        print(f"  ✗ structure_manipulation - {e}")
        return False

    try:
        from evaluation import SummarizationEvaluator
        print("  ✓ evaluation")

        # Quick test
        evaluator = SummarizationEvaluator()
        metrics = evaluator.compute_rouge(
            predictions=["This is a test."],
            references=["This is a test."]
        )
        assert 'rouge1' in metrics
        print("    ✓ Evaluation metrics working")

    except Exception as e:
        print(f"  ✗ evaluation - {e}")
        return False

    try:
        import config
        print("  ✓ config")
        print(f"    Models available: {list(config.MODELS['general'].keys()) + list(config.MODELS['medical'].keys())}")

    except Exception as e:
        print(f"  ✗ config - {e}")
        return False

    print("\n✓ All modules working!\n")
    return True


def test_model_loading():
    """Test loading a small model."""
    print("Testing model loading (this may take a moment)...")

    try:
        from models import ModelFactory

        # Try to load BART-CNN (if this fails, user needs to download it)
        print("  Loading BART-CNN...")
        model = ModelFactory.create_model('bart-cnn')

        # Quick inference test
        test_text = "Patient presented with chest pain and shortness of breath. Underwent cardiac evaluation."
        summary = model.summarize(test_text)

        print(f"  ✓ Model loaded and inference successful")
        print(f"    Test summary: {summary[:100]}...")

        return True

    except Exception as e:
        print(f"  ✗ Model loading failed: {e}")
        print("    This might be due to:")
        print("    - No internet connection (models auto-download from HuggingFace)")
        print("    - Insufficient disk space")
        print("    - GPU memory issues")
        return False


def test_data_availability():
    """Check if data file exists."""
    print("Checking data availability...")

    import os
    from config import DATA_PATH

    if os.path.exists(DATA_PATH):
        import pandas as pd
        df = pd.read_csv(DATA_PATH)
        print(f"  ✓ Data found: {len(df)} samples")

        # Check columns
        required_cols = ['text', 'reference_summary']
        missing = [col for col in required_cols if col not in df.columns]

        if missing:
            print(f"  ✗ Missing columns: {missing}")
            return False

        print(f"  ✓ All required columns present")
        return True

    else:
        print(f"  ✗ Data file not found: {DATA_PATH}")
        print("    Please place your MIMIC-IV dataset at this location")
        return False


def run_demo():
    """Run a complete mini demo."""
    print("\n" + "="*80)
    print("RUNNING MINI DEMO")
    print("="*80 + "\n")

    from structure_manipulation import NoteStructureManipulator
    from models import ModelFactory
    from evaluation import SummarizationEvaluator, format_metrics_table

    # Sample clinical note
    sample_note = """
Chief Complaint:
Chest pain

History of Present Illness:
The patient is a 65-year-old male who presented to the emergency
department with acute onset chest pain that started 2 hours ago.
Pain is substernal, radiating to left arm. Associated with diaphoresis.

Physical Exam:
Vitals: BP 145/90, HR 88, RR 16
General: Alert and oriented, in mild distress
Cardiovascular: Regular rate and rhythm, no murmurs

Brief Hospital Course:
Patient was evaluated and found to have ST elevation MI. Underwent
cardiac catheterization with stent placement to LAD. Post-procedure
course was uncomplicated. Discharged on day 3 with cardiac rehab.
    """

    reference = "65-year-old male with acute MI underwent cardiac catheterization with LAD stenting. Uncomplicated recovery, discharged day 3."

    # Create structure variants
    print("1. Creating structure variants...")
    manipulator = NoteStructureManipulator()

    original = manipulator.apply_variant(sample_note, 'original')
    no_formatting = manipulator.apply_variant(sample_note, 'no_formatting')

    print("  ✓ Created: original, no_formatting\n")

    # Load model
    print("2. Loading BART-CNN model...")
    model = ModelFactory.create_model('bart-cnn')
    print()

    # Generate summaries
    print("3. Generating summaries...")
    summary_original = model.summarize(original)
    summary_no_formatting = model.summarize(no_formatting)

    print(f"\n  Original structure summary:\n    {summary_original}")
    print(f"\n  No formatting summary:\n    {summary_no_formatting}\n")

    # Evaluate
    print("4. Evaluating...")
    evaluator = SummarizationEvaluator()

    metrics_original = evaluator.compute_all_metrics(
        [summary_original],
        [reference],
        include_bertscore=False
    )

    metrics_no_formatting = evaluator.compute_all_metrics(
        [summary_no_formatting],
        [reference],
        include_bertscore=False
    )

    print("\n  ORIGINAL STRUCTURE:")
    print(f"    ROUGE-1: {metrics_original['rouge1']:.4f}")
    print(f"    ROUGE-2: {metrics_original['rouge2']:.4f}")
    print(f"    ROUGE-L: {metrics_original['rougeL']:.4f}")

    print("\n  NO FORMATTING:")
    print(f"    ROUGE-1: {metrics_no_formatting['rouge1']:.4f}")
    print(f"    ROUGE-2: {metrics_no_formatting['rouge2']:.4f}")
    print(f"    ROUGE-L: {metrics_no_formatting['rougeL']:.4f}")

    # Calculate impact
    rouge1_drop = ((metrics_original['rouge1'] - metrics_no_formatting['rouge1'])
                   / metrics_original['rouge1'] * 100)

    print(f"\n  Structure Impact: {rouge1_drop:.1f}% drop in ROUGE-1 when formatting removed")

    print("\n" + "="*80)
    print("✓ DEMO COMPLETE - Setup is working correctly!")
    print("="*80)


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("MS THESIS SETUP VERIFICATION")
    print("RQ1: Effect of Note Structure on Clinical Summarization")
    print("="*80 + "\n")

    all_passed = True

    # Test imports
    if not test_imports():
        all_passed = False
        print("\n✗ Some imports failed. Please install missing packages.")
        print("Run: pip install -r requirements.txt\n")
        return

    # Test modules
    if not test_modules():
        all_passed = False

    # Test data
    data_ok = test_data_availability()

    # Test model (optional - can be slow)
    if '--skip-model' not in sys.argv:
        if not test_model_loading():
            print("\n  Note: Model loading failed, but you can still run experiments")
            print("  The model will be downloaded automatically when you run the experiment\n")

    if all_passed:
        print("\n" + "="*80)
        print("✓ ALL TESTS PASSED!")
        print("="*80)

        if data_ok and '--demo' in sys.argv:
            run_demo()
        else:
            print("\nYou're ready to run experiments!")
            print("\nNext steps:")
            print("  1. Quick test:  python experiment.py")
            print("  2. Full run:    python experiment.py --full")
            print("  3. Visualize:   python visualize.py")
            print("\nFor a complete demo, run: python test_setup.py --demo")

    else:
        print("\n" + "="*80)
        print("✗ SOME TESTS FAILED")
        print("="*80)
        print("\nPlease fix the issues above before running experiments.")


if __name__ == '__main__':
    main()
