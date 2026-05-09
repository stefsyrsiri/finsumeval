from statistics import harmonic_mean

import torch
from loguru import logger
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, pipeline, ZeroShotClassificationPipeline

from metric.tokenizers.tokenizer import SpacyTokenizer
from metric.evaluators.base import Evaluator
from metric.evaluators.schemas import ConcisenessScore, FaithfulnessScore, SumEvalScore
from metric.evaluators.sentence_models_registry import SENTENCE_MODELS
from metric.evaluators.utils import classify_with_zero_shot_model, find_redundant_pairs


class SumEval(Evaluator):
    def __init__(
        self,
        lang: str = "en",
        nli_model: str = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
    ):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.lang = lang
        self.nli_model = nli_model

        self.tokenizer = SpacyTokenizer(lang)
        self.smodel = SentenceTransformer(SENTENCE_MODELS.get(lang), device=device)
        self.nli_clf = self._load_nli_model(device)

    def _load_nli_model(self, device: str) -> ZeroShotClassificationPipeline:
        """Loads a zero-shot classification pipeline for NLI tasks."""
        tokenizer = AutoTokenizer.from_pretrained(self.nli_model, use_fast=True)
        return pipeline(task="zero-shot-classification", model=self.nli_model, tokenizer=tokenizer, device=device)

    def score_faithfulness(
        self,
        summary: str,
        source: str,
        threshold: float = 0.55,
        labels: list[str] = ["entailment", "neutral", "contradiction"],
    ) -> FaithfulnessScore:
        """Scores the faithfulness of the summary with respect to the source text.

        Args:
            summary (str): The candidate summary to evaluate.
            source (str): The source text.
            threshold (float): The entailment score threshold to consider a statement as faithful. Defaults to 0.55.
        Returns:
            FaithfulnessScore: An object containing the faithfulness score and list of unfaithful statements.
        """
        source_nlp = self.tokenizer(source)
        src_statements = self.tokenizer.sentencize(source_nlp)
        logger.debug(f"Source statements: {len(src_statements)}")

        summ_nlp = self.tokenizer(summary)
        summ_statements = self.tokenizer.sentencize(summ_nlp)
        logger.debug(f"Summary statements: {len(summ_statements)}")

        unfaithful_indices = set()
        used_src_indices = set()

        nli_calls = 0
        # Check each summary statement
        for summ_idx, summ_statement in enumerate(summ_statements):
            # Against each source statement
            for src_idx, src_statement in enumerate(src_statements):
                # Skip already matched source statements
                if src_idx in used_src_indices:
                    continue
                entailment_score = classify_with_zero_shot_model(
                    model=self.nli_clf,
                    labels=labels,
                    src_statement=src_statement,
                    summ_statement=summ_statement,
                    target="entailment",
                )
                nli_calls += 1

                # If at least one entailment is found, the statement is faithful
                if entailment_score > threshold:
                    # Mark source statement as used
                    used_src_indices.add(src_idx)
                    break

                # If all source statements are checked and none entail the summary statement, the statement is unfaithful
                if src_statement == src_statements[-1]:
                    unfaithful_indices.add(summ_idx)

        logger.debug(f"NLI calls made: {nli_calls}")
        faithfulness = 1 - len(unfaithful_indices) / len(summ_statements)
        return FaithfulnessScore(score=faithfulness, unfaithful_statements=list(unfaithful_indices))

    def score_conciseness(self, summary: str, threshold: float = 0.9) -> ConcisenessScore:
        """Scores the conciseness of the summary.

        Args:
            summary (str): The candidate summary to evaluate.
            threshold (float): The cosine similarity threshold to consider statements as redundant.
        Returns:
            ConcisenessScore: An object containing the conciseness score and list of redundant statements.
        """
        if not isinstance(summary, str) or summary.strip() == "":
            raise ValueError("The input summary must not be empty.")

        summ_nlp = self.tokenizer(summary)
        summ_statements = self.tokenizer.sentencize(summ_nlp)
        logger.debug(f"Summary statements: {len(summ_statements)}")

        # Encode summary statements
        summ_embeddings = self.smodel.encode(summ_statements, convert_to_tensor=True)

        # Compute cosine similarity matrix
        cosine_scores = util.pytorch_cos_sim(summ_embeddings, summ_embeddings)

        # Identify redundant pairs
        unique_statements, redundant_pairs = find_redundant_pairs(
            statements=summ_statements, similarity_scores=cosine_scores, similarity_threshold=threshold
        )
        conciseness_score = len(unique_statements) / len(summ_statements)
        return ConcisenessScore(score=conciseness_score, redundant_statements=list(redundant_pairs))

    def score(
        self, summary: str, source: str, faithfulness_threshold: float = 0.55, conciseness_threshold: float = 0.9
    ) -> SumEvalScore:
        """Scores the summary based on faithfulness and conciseness.

        Args:
            summary (str): The summary text.
            source (str): The source text.
            faithfulness_threshold (float): The entailment score threshold for faithfulness. Defaults to 0.55.
            conciseness_threshold (float): The cosine similarity threshold for conciseness. Defaults to 0.9.
        Returns:
            SumEvalScore: An object containing the overall score, faithfulness, and conciseness.
        """
        faithfulness = self.score_faithfulness(summary, source, faithfulness_threshold)
        conciseness = self.score_conciseness(summary, conciseness_threshold)
        final_score = (
            harmonic_mean([faithfulness.score, conciseness.score])
            if faithfulness.score > 0 and conciseness.score > 0
            else 0.0
        )
        return SumEvalScore(score=final_score, faithfulness=faithfulness, conciseness=conciseness)
