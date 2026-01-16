"""
Configuration for Refined RQ1: Effect of Note Structure Across Training Paradigms

NEW DESIGN:
- Type 0: Domain Pre-trained Only (BioBART) - Negative Control
- Type 1: Task-Specific Summarization (BART-CNN, PEGASUS)
- Type 2: Instruction-Tuned (Flan-T5, SciFive/ClinicalT5)
"""

# ============================================================================
# Model Configuration
# ============================================================================

MODELS = {
    # Type 0: Domain Pre-trained (No Task Tuning) - NEGATIVE CONTROL
    'type0_domain_only': {
        'biobart': 'GanjinZero/biobart-v2-large',  # Shown to produce garbage
    },

    # Type 1: Task-Specific Summarization Models
    'type1_task_specific': {
        'bart-cnn': 'facebook/bart-large-cnn',      # General summarization
        'pegasus': 'google/pegasus-large',          # General summarization
    },

    # Type 2: Instruction-Tuned Models
    'type2_instruction_tuned': {
        # General instruction-tuned
        'flan-t5-xl': 'google/flan-t5-xl',          # 3B params - RECOMMENDED
        'flan-t5-large': 'google/flan-t5-large',    # 780M params - if memory constrained
        'flan-t5-xxl': 'google/flan-t5-xxl',        # 11B params - if A100 available

        # Medical instruction-tuned
        'scifive': 'razent/SciFive-large-Pubmed_PMC',       # PubMed/PMC, native PyTorch
        # 'clinical-t5': 'luqh/ClinicalT5-large',           # MIMIC-III (Flax - may need conversion)
    },
}

# Flattened model list for easy iteration (RECOMMENDED SET)
RECOMMENDED_MODELS = {
    # Minimum viable set (3 models)
    'bart-cnn': 'facebook/bart-large-cnn',          # Type 1 baseline
    'flan-t5-xl': 'google/flan-t5-xl',             # Type 2 general
    'scifive': 'razent/SciFive-large-Pubmed_PMC',  # Type 2 medical
}

# Full model set including negative control (5 models)
FULL_MODELS = {
    'biobart': 'GanjinZero/biobart-v2-large',      # Type 0 (negative control)
    'bart-cnn': 'facebook/bart-large-cnn',          # Type 1
    'pegasus': 'google/pegasus-large',              # Type 1
    'flan-t5-xl': 'google/flan-t5-xl',             # Type 2 general
    'scifive': 'razent/SciFive-large-Pubmed_PMC',  # Type 2 medical
}

# Memory-constrained set (use if GPU < 24GB)
MEMORY_EFFICIENT_MODELS = {
    'bart-cnn': 'facebook/bart-large-cnn',          # ~1.5GB
    'flan-t5-large': 'google/flan-t5-large',       # ~3GB (instead of XL)
    'scifive': 'razent/SciFive-large-Pubmed_PMC',  # ~3GB
}

# Active model set (CHANGE THIS BASED ON YOUR CHOICE)
ACTIVE_MODELS = RECOMMENDED_MODELS  # or FULL_MODELS or MEMORY_EFFICIENT_MODELS

# ============================================================================
# Model-Specific Prompts (Instruction-Tuned Models)
# ============================================================================

# Flan-T5 uses "task: input" format
INSTRUCTION_PROMPTS = {
    'flan-t5-xl': 'summarize: {}',
    'flan-t5-large': 'summarize: {}',
    'flan-t5-xxl': 'summarize: {}',
    'scifive': 'summarize: {}',  # SciFive also uses T5 format
}

# More explicit prompt (optional - test which works better)
INSTRUCTION_PROMPTS_EXPLICIT = {
    'flan-t5-xl': 'summarize the following clinical note: {}',
    'flan-t5-large': 'summarize the following clinical note: {}',
    'flan-t5-xxl': 'summarize the following clinical note: {}',
    'scifive': 'summarize the following clinical discharge summary: {}',
}

# Default: use simple prompts (works well for Flan-T5)
ACTIVE_PROMPTS = INSTRUCTION_PROMPTS

# ============================================================================
# Generation Parameters
# ============================================================================

# Default generation config (works for most models)
GENERATION_CONFIG = {
    'max_length': 250,        # Allow up to 250 tokens
    'min_length': 100,        # Minimum 100 tokens
    'num_beams': 4,           # Beam search
    'length_penalty': 0.8,    # Slight penalty for length
    'early_stopping': True,
    'no_repeat_ngram_size': 3,
}

# Model-specific generation configs (if needed)
MODEL_SPECIFIC_GENERATION = {
    'flan-t5-xl': {
        'max_length': 256,     # T5 works better with powers of 2
        'min_length': 100,
        'num_beams': 4,
        'length_penalty': 1.0,  # Flan-T5 handles length well
        'early_stopping': True,
        'no_repeat_ngram_size': 3,
    },
    'scifive': {
        'max_length': 256,
        'min_length': 100,
        'num_beams': 4,
        'length_penalty': 0.8,
        'early_stopping': True,
        'no_repeat_ngram_size': 3,
    },
}

# ============================================================================
# Input Processing
# ============================================================================

# Model-specific max input lengths (based on architecture)
MAX_INPUT_LENGTHS = {
    'bart-cnn': 1024,      # BART max position embeddings
    'pegasus': 1024,       # PEGASUS max
    'biobart': 1024,       # BioBART max
    'flan-t5-xl': 512,     # T5 typically 512 (can extend to 1024)
    'flan-t5-large': 512,
    'flan-t5-xxl': 512,
    'scifive': 512,
}

# Default max input length
MAX_INPUT_LENGTH = 1024  # Will be overridden per model

# ============================================================================
# Structure Variants
# ============================================================================

STRUCTURE_VARIANTS = [
    'original',      # Keep original structure with headers
    'no_headers',    # Remove section headers
    'no_formatting', # Remove all formatting (headers, bullets, newlines)
    'shuffled',      # Shuffle sections to disrupt logical flow
]

# ============================================================================
# Evaluation Metrics
# ============================================================================

METRICS = [
    'rouge1',
    'rouge2',
    'rougeL',
    'bertscore',
]

# ============================================================================
# Dataset Configuration
# ============================================================================

DATA_PATH = 'mimic_1000_notes_with_references.csv'
SAMPLE_SIZE = 1000  # Use all samples

# ============================================================================
# Output Configuration
# ============================================================================

OUTPUT_DIR = 'results_v2'           # New results directory
CHECKPOINT_DIR = 'checkpoints_v2'
PREDICTIONS_DIR = 'predictions_v2'  # For reference-free analysis

# ============================================================================
# Experiment Settings
# ============================================================================

RANDOM_SEED = 42
BATCH_SIZE = 4  # Adjust based on GPU memory

# ============================================================================
# Helper Functions
# ============================================================================

def get_model_prompt(model_name, text):
    """Get the appropriate prompt for a model."""
    if model_name in ACTIVE_PROMPTS:
        return ACTIVE_PROMPTS[model_name].format(text)
    else:
        # No prompt for task-specific models
        return text

def get_model_max_length(model_name):
    """Get max input length for a model."""
    return MAX_INPUT_LENGTHS.get(model_name, MAX_INPUT_LENGTH)

def get_model_generation_config(model_name):
    """Get generation config for a model."""
    return MODEL_SPECIFIC_GENERATION.get(model_name, GENERATION_CONFIG)

def get_model_type(model_name):
    """Get the training paradigm type for a model."""
    if model_name in ['biobart']:
        return 'Type 0: Domain Pre-trained Only'
    elif model_name in ['bart-cnn', 'pegasus']:
        return 'Type 1: Task-Specific'
    elif model_name in ['flan-t5-xl', 'flan-t5-large', 'flan-t5-xxl', 'scifive']:
        return 'Type 2: Instruction-Tuned'
    else:
        return 'Unknown'

# ============================================================================
# Experimental Hypotheses
# ============================================================================

EXPECTED_ROBUSTNESS_RANKING = [
    '1. Type 2 (Instruction-Tuned) - Most robust to structure disruption',
    '2. Type 1 (Task-Specific) - Moderately robust',
    '3. Type 0 (Domain Only) - Least robust / produces garbage'
]

EXPECTED_SHUFFLED_DROPS = {
    'Type 0 (biobart)': '-43% (already observed)',
    'Type 1 (bart-cnn)': '-12% (already observed)',
    'Type 2 (flan-t5)': '-5% to -10% (predicted - more robust)',
}

# ============================================================================
# Usage Examples
# ============================================================================

if __name__ == '__main__':
    print("\nActive Models Configuration")
    print("="*60)

    for model_name, model_id in ACTIVE_MODELS.items():
        model_type = get_model_type(model_name)
        max_len = get_model_max_length(model_name)

        print(f"\n{model_name}:")
        print(f"  Type: {model_type}")
        print(f"  HuggingFace ID: {model_id}")
        print(f"  Max Input: {max_len} tokens")

        # Show prompt if applicable
        if model_name in ACTIVE_PROMPTS:
            sample_text = "[clinical note]"
            prompt = get_model_prompt(model_name, sample_text)
            print(f"  Prompt: {prompt}")

    print("\n" + "="*60)
    print(f"\nTotal models: {len(ACTIVE_MODELS)}")
    print(f"Structure variants: {len(STRUCTURE_VARIANTS)}")
    print(f"Total experiments: {len(ACTIVE_MODELS) * len(STRUCTURE_VARIANTS)}")
    print("="*60)
