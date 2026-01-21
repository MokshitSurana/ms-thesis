"""
RQ2: Prompt Engineering for Instruction-Tuned Models
Tests how different prompts affect summarization behavior.
"""

from config_v2 import *

# Override prompts for RQ2
PROMPT_VARIANTS = {
    'simple': 'summarize: {}',
    'explicit': 'summarize the following clinical discharge note: {}',
    'targeted': 'extract the key clinical findings and create a brief summary: {}',
    'constrained': 'summarize this clinical note in 2-3 sentences focusing on diagnosis and treatment: {}',
    'roleplay': 'You are a clinical summarization assistant. Provide a concise summary of this patient encounter: {}',
    'extractive': 'identify and list the most important clinical information from this note: {}',
}

# Only test on instruction-tuned models
ACTIVE_MODELS_RQ2 = {
    'flan-t5-xl': 'google/flan-t5-xl',
    'scifive': 'razent/SciFive-large-Pubmed_PMC',
}

# Test on original structure only (to isolate prompt effect)
STRUCTURE_VARIANTS_RQ2 = ['original']

OUTPUT_DIR_RQ2 = 'results_rq2_prompts'
PREDICTIONS_DIR_RQ2 = 'predictions_rq2_prompts'

# Expected findings:
# - 'simple' → baseline
# - 'constrained' → shorter, more focused summaries
# - 'roleplay' → more natural language
# - 'extractive' → bulleted lists?
# - 'targeted' → better medical term coverage

if __name__ == '__main__':
    print("RQ2: Prompt Engineering Experiment")
    print(f"Models: {list(ACTIVE_MODELS_RQ2.keys())}")
    print(f"Prompts: {list(PROMPT_VARIANTS.keys())}")
    print(f"Total experiments: {len(ACTIVE_MODELS_RQ2) * len(PROMPT_VARIANTS)}")
