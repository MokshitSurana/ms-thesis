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
        'clinical-t5': 'luqh/ClinicalT5-large',  # Medical T5 variant
        'biogpt': 'microsoft/biogpt-large',       # Alternative medical model
    }
}

# Generation Parameters
GENERATION_CONFIG = {
    'max_length': 150,
    'min_length': 50,
    'num_beams': 4,
    'length_penalty': 2.0,
    'early_stopping': True,
    'no_repeat_ngram_size': 3,
}

# Input Processing
MAX_INPUT_LENGTH = 1024  # Max tokens for input

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
