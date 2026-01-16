"""
Test Model Compatibility
Checks which instruction-tuned models can load on your system.
"""

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import sys
from pathlib import Path

def get_gpu_info():
    """Get GPU information."""
    if torch.cuda.is_available():
        print("GPU Information:")
        print(f"  Device: {torch.cuda.get_device_name(0)}")
        print(f"  Total Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"  Available Memory: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB allocated")
        print()
        return True
    else:
        print("⚠️ No GPU detected! Models will run on CPU (very slow)")
        print()
        return False

def test_model_loading(model_name, model_id, test_generation=True):
    """Test if a model can be loaded and used."""
    print(f"\n{'='*70}")
    print(f"Testing: {model_name}")
    print(f"HuggingFace ID: {model_id}")
    print('='*70)

    try:
        # Load tokenizer
        print("  Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        print("  ✓ Tokenizer loaded")

        # Load model
        print("  Loading model...")
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map='auto' if torch.cuda.is_available() else None
        )
        print("  ✓ Model loaded")

        # Test tokenization
        sample_text = "summarize: The patient was admitted with chest pain and shortness of breath."
        print(f"\n  Testing tokenization...")
        print(f"  Input: '{sample_text[:60]}...'")

        inputs = tokenizer(sample_text, return_tensors='pt', max_length=512, truncation=True)

        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        print(f"  ✓ Tokenized: {inputs['input_ids'].shape[1]} tokens")

        # Test generation
        if test_generation:
            print(f"\n  Testing generation...")
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=50,
                    num_beams=2,
                    early_stopping=True
                )

            decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"  ✓ Generated: '{decoded}'")

        # Memory usage
        if torch.cuda.is_available():
            memory_used = torch.cuda.max_memory_allocated(0) / 1e9
            print(f"\n  GPU Memory Used: {memory_used:.2f} GB")
            torch.cuda.reset_peak_memory_stats()

        # Cleanup
        del model
        del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"\n✓✓✓ {model_name} is COMPATIBLE ✓✓✓")
        return True

    except Exception as e:
        print(f"\n✗✗✗ {model_name} FAILED ✗✗✗")
        print(f"  Error: {str(e)}")
        print(f"  Error type: {type(e).__name__}")

        # Cleanup on error
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return False


def main():
    print("\n" + "="*70)
    print("Model Compatibility Test for Instruction-Tuned Models")
    print("="*70 + "\n")

    # Check GPU
    has_gpu = get_gpu_info()

    # Models to test
    models_to_test = {
        # Instruction-tuned (general)
        'flan-t5-large': 'google/flan-t5-large',        # 780M params (~3GB)
        'flan-t5-xl': 'google/flan-t5-xl',              # 3B params (~12GB)

        # Instruction-tuned (medical)
        'scifive': 'razent/SciFive-large-Pubmed_PMC',   # ~3GB

        # OPTIONAL: Only test if you have A100 (40GB+)
        # 'flan-t5-xxl': 'google/flan-t5-xxl',          # 11B params (~44GB)

        # OPTIONAL: Test ClinicalT5 (may have Flax issues)
        # 'clinical-t5': 'luqh/ClinicalT5-large',
    }

    results = {}

    for model_name, model_id in models_to_test.items():
        success = test_model_loading(model_name, model_id, test_generation=True)
        results[model_name] = success

        # Wait for user to see results
        print("\n" + "-"*70)
        print("Press Enter to continue to next model (or Ctrl+C to stop)...")
        try:
            input()
        except KeyboardInterrupt:
            print("\n\nStopped by user.")
            break

    # Summary
    print("\n" + "="*70)
    print("COMPATIBILITY SUMMARY")
    print("="*70)

    compatible = []
    incompatible = []

    for model_name, success in results.items():
        if success:
            print(f"  ✓ {model_name}")
            compatible.append(model_name)
        else:
            print(f"  ✗ {model_name}")
            incompatible.append(model_name)

    print("\n" + "="*70)
    print(f"Compatible: {len(compatible)}/{len(results)}")

    if compatible:
        print("\nRECOMMENDED MODEL SET:")
        print("  Update config_v2.py with:")
        print("  ACTIVE_MODELS = {")
        for model_name in compatible:
            model_id = models_to_test[model_name]
            print(f"      '{model_name}': '{model_id}',")
        print("  }")

    if incompatible:
        print(f"\n⚠️ Incompatible models: {', '.join(incompatible)}")
        print("  Try:")
        print("  1. Use smaller variants (flan-t5-large instead of xl)")
        print("  2. Reduce batch size in config_v2.py")
        print("  3. Use gradient checkpointing (add to models.py)")

    print("="*70 + "\n")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
