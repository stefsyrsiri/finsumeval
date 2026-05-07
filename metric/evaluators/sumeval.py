from statistics import harmonic_mean

import torch
from loguru import logger
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, pipeline, ZeroShotClassificationPipeline

from metric.tokenizers.tokenizer import Tokenizer
from metric.evaluators.schemas import ConcisenessScore, FaithfulnessScore, SumEvalScore
from metric.evaluators.sentence_models_registry import SMODELS


class SumEval:
    def __init__(
        self,
        lang: str = "en",
        nli_model: str = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
    ):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.lang = lang
        self.nli_model = nli_model

        self.tokenizer = Tokenizer(lang)
        self.smodel = SentenceTransformer(SMODELS.get(lang), device=device)
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
    ) -> FaithfulnessScore:
        """Scores the faithfulness of the summary with respect to the source text.

        Args:
            summary (str): The candidate summary to evaluate.
            source (str): The source text.
            threshold (float): The entailment score threshold to consider a statement as faithful. Defaults to 0.55.
        Returns:
            FaithfulnessScore: An object containing the faithfulness score and list of unfaithful statements.
        """
        LABELS = ["entailment", "neutral", "contradiction"]
        source_nlp = self.tokenizer(source)
        src_statements = self.tokenizer.sentencize(source_nlp)
        logger.debug(f"Source statements: {len(src_statements)}")

        summ_nlp = self.tokenizer(summary)
        summ_statements = self.tokenizer.sentencize(summ_nlp)
        logger.debug(f"Summary statements: {len(summ_statements)}")

        unfaithful_indices = set()
        ent_src_indices = {}
        src_indices = list(range(len(src_statements)))
        summ_indices = list(range(len(summ_statements)))

        nli_calls = 0
        # Check each summary statement
        for summ_statement in summ_statements:
            # Against each source statement
            for src_statement in src_statements:
                sequence_to_classify = f"reference: {src_statement}\n\ncandidate: {summ_statement}"
                classification = self.nli_clf(sequence_to_classify, LABELS)
                nli_calls += 1
                entailment_score = classification["scores"][classification["labels"].index("entailment")]

                # If at least one entailment is found, the statement is faithful
                if entailment_score > threshold:
                    key = src_indices[src_statements.index(src_statement)]
                    if key in ent_src_indices:
                        ent_src_indices[key].append(summ_indices[summ_statements.index(summ_statement)])
                    else:
                        ent_src_indices[key] = [summ_indices[summ_statements.index(summ_statement)]]
                    break

                # If all source statements are checked and none entail the summary statement, the statement is unfaithful
                if src_statement == src_statements[-1]:
                    key = summ_indices[summ_statements.index(summ_statement)]
                    unfaithful_indices.add(key)

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

        redundant_pairs = []
        unique_indices = set(range(len(summ_statements)))

        # Identify redundant pairs
        for i in range(len(summ_statements)):
            for j in range(i + 1, len(summ_statements)):
                if cosine_scores[i][j] > threshold:
                    redundant_pairs.append((i, j))
                    if j in unique_indices:
                        unique_indices.remove(j)

        conciseness_score = len(unique_indices) / len(summ_statements)

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

        return SumEvalScore(
            score=final_score,
            faithfulness=faithfulness,
            conciseness=conciseness,
        )
