"""
Configuration for RQ1: Effect of Note Structure on Clinical Summarization
"""

# Models Configuration
MODELS = {
    'general': {
        'bart-cnn': 'facebook/bart-large-cnn',
        'pegasus': 'google/pegasus-large',
    },
    'medical': {
        # Native PyTorch models (no flax dependency)
        'biobart': 'GanjinZero/biobart-v2-large',      # BioBERT + BART
        'clinical-longformer': 'yikuan8/Clinical-Longformer',  # Long clinical docs
        # Note: Clinical-T5 requires flax (dependency conflicts with CUDA)
        # 'clinical-t5': 'luqh/ClinicalT5-large',
    }
}

# Generation Parameters
GENERATION_CONFIG = {
    'max_length': 250,        # Increased from 150 to allow longer summaries (~185 words)
    'min_length': 100,        # Increased from 50 to ensure detail
    'num_beams': 4,
    'length_penalty': 0.8,    # Reduced from 2.0 to encourage longer output
    'early_stopping': True,
    'no_repeat_ngram_size': 3,
}

# Input Processing
MAX_INPUT_LENGTH = 1024  # BART max position embeddings (cannot exceed without model modification)

# Note Structure Variants
STRUCTURE_VARIANTS = [
    'original',      # Keep original structure with headers
    'no_headers',    # Remove section headers
    'no_formatting', # Remove all formatting (headers, bullets, newlines)
    'shuffled',      # Shuffle sections to disrupt logical flow
]

# Evaluation Metrics
METRICS = [
    'rouge1',
    'rouge2',
    'rougeL',
    'bertscore',
]

# Dataset Configuration
DATA_PATH = 'mimic_1000_notes_with_references.csv'
SAMPLE_SIZE = 1000  # Use all samples, or reduce for testing

# Output Configuration
OUTPUT_DIR = 'results'
CHECKPOINT_DIR = 'checkpoints'

# Experiment Settings
RANDOM_SEED = 42
BATCH_SIZE = 4  # Adjust based on GPU memory
