"""
Model wrapper for general and medical summarization models.
"""

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from typing import List, Dict, Optional
from config import GENERATION_CONFIG, MAX_INPUT_LENGTH


class SummarizationModel:
    """
    Wrapper for summarization models (general or medical).
    """

    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        generation_config: Optional[Dict] = None
    ):
        """
        Initialize model and tokenizer.

        Args:
            model_name: HuggingFace model identifier
            device: 'cuda' or 'cpu', auto-detected if None
            generation_config: Custom generation parameters
        """
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        print(f"Loading {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model = self.model.to(self.device)
        self.model.eval()

        self.generation_config = generation_config or GENERATION_CONFIG
        print(f"✓ Model loaded on {self.device}")

    def summarize(self, text: str, **generation_kwargs) -> str:
        """
        Generate summary for a single text.

        Args:
            text: Input clinical note
            **generation_kwargs: Override default generation config

        Returns:
            Generated summary
        """
        # Merge default config with kwargs
        gen_config = {**self.generation_config, **generation_kwargs}

        # Tokenize
        inputs = self.tokenizer(
            text,
            max_length=MAX_INPUT_LENGTH,
            truncation=True,
            return_tensors='pt'
        ).to(self.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_config)

        # Decode
        summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return summary

    def summarize_batch(self, texts: List[str], **generation_kwargs) -> List[str]:
        """
        Generate summaries for a batch of texts.

        Args:
            texts: List of input clinical notes
            **generation_kwargs: Override default generation config

        Returns:
            List of generated summaries
        """
        gen_config = {**self.generation_config, **generation_kwargs}

        # Tokenize batch
        inputs = self.tokenizer(
            texts,
            max_length=MAX_INPUT_LENGTH,
            truncation=True,
            padding=True,
            return_tensors='pt'
        ).to(self.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_config)

        # Decode all
        summaries = [
            self.tokenizer.decode(output, skip_special_tokens=True)
            for output in outputs
        ]

        return summaries

    def __repr__(self):
        return f"SummarizationModel({self.model_name}, device={self.device})"


class ModelFactory:
    """
    Factory for creating model instances.
    """

    @staticmethod
    def create_model(model_key: str, device: Optional[str] = None) -> SummarizationModel:
        """
        Create model from config key.

        Args:
            model_key: Key from MODELS config (e.g., 'bart-cnn', 'clinical-t5')
            device: Device to load model on

        Returns:
            SummarizationModel instance
        """
        from config import MODELS

        # Search in both general and medical models
        for category in ['general', 'medical']:
            if model_key in MODELS[category]:
                model_name = MODELS[category][model_key]
                return SummarizationModel(model_name, device=device)

        raise ValueError(
            f"Unknown model key: {model_key}. "
            f"Available: {list(MODELS['general'].keys()) + list(MODELS['medical'].keys())}"
        )


def test_model():
    """
    Quick test of model functionality.
    """
    sample_text = """
    Chief Complaint: Patient with chest pain.

    History of Present Illness: The patient is a 65-year-old male who
    presented to the emergency department with acute onset chest pain
    that started 2 hours ago. Pain is substernal, radiating to left arm.

    Physical Exam: BP 145/90, HR 88, alert and oriented.

    Brief Hospital Course: Patient was evaluated and found to have
    ST elevation MI. Cardiac catheterization performed with stent placement.
    Patient recovered well and was discharged on day 3.
    """

    print("Testing BART-CNN model...")
    model = ModelFactory.create_model('bart-cnn')
    summary = model.summarize(sample_text)

    print("\nInput:")
    print(sample_text[:200] + "...")
    print("\nGenerated Summary:")
    print(summary)


if __name__ == '__main__':
    test_model()
