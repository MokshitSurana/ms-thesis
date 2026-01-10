"""
Evaluation metrics for clinical summarization.
Implements ROUGE, BERTScore, and custom medical metrics.
"""

import numpy as np
from typing import List, Dict, Tuple
from rouge_score import rouge_scorer
from bert_score import score as bertscore_compute


class SummarizationEvaluator:
    """
    Evaluate summarization quality using multiple metrics.
    """

    def __init__(self):
        # Initialize ROUGE scorer
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'],
            use_stemmer=True
        )

    def compute_rouge(
        self,
        predictions: List[str],
        references: List[str]
    ) -> Dict[str, float]:
        """
        Compute ROUGE scores for predictions vs references.

        Args:
            predictions: List of generated summaries
            references: List of reference summaries

        Returns:
            Dictionary with average ROUGE scores
        """
        assert len(predictions) == len(references), "Length mismatch"

        rouge1_scores = []
        rouge2_scores = []
        rougeL_scores = []

        for pred, ref in zip(predictions, references):
            scores = self.rouge_scorer.score(ref, pred)

            rouge1_scores.append(scores['rouge1'].fmeasure)
            rouge2_scores.append(scores['rouge2'].fmeasure)
            rougeL_scores.append(scores['rougeL'].fmeasure)

        return {
            'rouge1': np.mean(rouge1_scores),
            'rouge2': np.mean(rouge2_scores),
            'rougeL': np.mean(rougeL_scores),
            'rouge1_std': np.std(rouge1_scores),
            'rouge2_std': np.std(rouge2_scores),
            'rougeL_std': np.std(rougeL_scores),
        }

    def compute_bertscore(
        self,
        predictions: List[str],
        references: List[str],
        lang: str = 'en',
        model_type: str = 'microsoft/deberta-xlarge-mnli'
    ) -> Dict[str, float]:
        """
        Compute BERTScore for semantic similarity.

        Args:
            predictions: List of generated summaries
            references: List of reference summaries
            lang: Language code
            model_type: BERT model to use for scoring

        Returns:
            Dictionary with BERTScore metrics
        """
        P, R, F1 = bertscore_compute(
            predictions,
            references,
            lang=lang,
            model_type=model_type,
            verbose=False
        )

        return {
            'bertscore_precision': P.mean().item(),
            'bertscore_recall': R.mean().item(),
            'bertscore_f1': F1.mean().item(),
            'bertscore_f1_std': F1.std().item(),
        }

    def compute_all_metrics(
        self,
        predictions: List[str],
        references: List[str],
        include_bertscore: bool = True
    ) -> Dict[str, float]:
        """
        Compute all evaluation metrics.

        Args:
            predictions: Generated summaries
            references: Reference summaries
            include_bertscore: Whether to compute BERTScore (slower)

        Returns:
            Dictionary with all metrics
        """
        metrics = {}

        # ROUGE scores
        print("Computing ROUGE scores...")
        rouge_scores = self.compute_rouge(predictions, references)
        metrics.update(rouge_scores)

        # BERTScore
        if include_bertscore:
            print("Computing BERTScore (this may take a while)...")
            bert_scores = self.compute_bertscore(predictions, references)
            metrics.update(bert_scores)

        return metrics

    def compute_length_stats(
        self,
        predictions: List[str],
        references: List[str]
    ) -> Dict[str, float]:
        """
        Compute length statistics for summaries.

        Returns:
            Dictionary with length metrics
        """
        pred_lengths = [len(p.split()) for p in predictions]
        ref_lengths = [len(r.split()) for r in references]

        return {
            'pred_avg_length': np.mean(pred_lengths),
            'pred_std_length': np.std(pred_lengths),
            'ref_avg_length': np.mean(ref_lengths),
            'ref_std_length': np.std(ref_lengths),
        }


def format_metrics_table(metrics: Dict[str, float]) -> str:
    """
    Format metrics as a nice table string.
    """
    lines = ["=" * 60]
    lines.append("EVALUATION METRICS")
    lines.append("=" * 60)

    # Group metrics
    rouge_metrics = {k: v for k, v in metrics.items() if 'rouge' in k.lower()}
    bert_metrics = {k: v for k, v in metrics.items() if 'bert' in k.lower()}
    length_metrics = {k: v for k, v in metrics.items() if 'length' in k.lower()}

    if rouge_metrics:
        lines.append("\nROUGE Scores:")
        lines.append("-" * 60)
        for k, v in rouge_metrics.items():
            lines.append(f"  {k:20s}: {v:.4f}")

    if bert_metrics:
        lines.append("\nBERTScore:")
        lines.append("-" * 60)
        for k, v in bert_metrics.items():
            lines.append(f"  {k:20s}: {v:.4f}")

    if length_metrics:
        lines.append("\nLength Statistics:")
        lines.append("-" * 60)
        for k, v in length_metrics.items():
            lines.append(f"  {k:20s}: {v:.2f}")

    lines.append("=" * 60)

    return "\n".join(lines)


def demo_evaluation():
    """
    Demonstrate evaluation with example data.
    """
    predictions = [
        "Patient presented with chest pain and was diagnosed with MI. Underwent stenting.",
        "The patient had pneumonia and was treated with antibiotics for 5 days.",
    ]

    references = [
        "65-year-old male with acute MI underwent cardiac catheterization with stent placement.",
        "Patient admitted with community-acquired pneumonia, treated with IV antibiotics.",
    ]

    evaluator = SummarizationEvaluator()

    # Compute metrics
    metrics = evaluator.compute_all_metrics(predictions, references, include_bertscore=False)
    length_stats = evaluator.compute_length_stats(predictions, references)
    metrics.update(length_stats)

    # Print formatted results
    print(format_metrics_table(metrics))


if __name__ == '__main__':
    demo_evaluation()
