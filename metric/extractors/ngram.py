from collections import Counter
from dataclasses import dataclass
from time import perf_counter

from loguru import logger


@dataclass
class SentenceData:
    sentence: str
    ngrams: Counter


@dataclass
class MatchedSentence:
    summary_sentence: str
    best_sentence: str
    best_score: float


class NgramExtractor:
    """Extracts sentences from the source document that have n-gram overlap with the summary sentences."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def _get_sentence_data(self, text: str) -> dict:
        """Get sentences and ngrams"""
        sentence_dict = {}
        nlp = self.tokenizer(text.strip())
        for i, sent in enumerate(nlp.sents):
            tokens = [token.lemma_.lower() for token in sent if not token.is_stop and not token.is_punct]
            ngrams = Counter(tokens)
            sentence_dict[i] = SentenceData(sentence=sent.text, ngrams=ngrams)
        return sentence_dict

    def _ngram_overlap(self, summary_ngrams: Counter, source_ngrams: Counter) -> float:
        """Compute ngram overlap"""
        total_summary_ngrams = sum(summary_ngrams.values())
        overlap = sum((summary_ngrams & source_ngrams).values())
        return overlap / total_summary_ngrams if total_summary_ngrams > 0 else 0

    def _find_matching_sentences(
        self, source_sentence_dict: dict, summary_sentence_dict: dict
    ) -> list[MatchedSentence]:
        """Find sentences with good overlap"""
        best_matches = []
        used_indices = set()

        for summary_sentence in summary_sentence_dict.values():
            best_score = -1
            best_sentence_idx = None

            for source_sentence_idx, source_sentence in source_sentence_dict.items():
                if source_sentence_idx in used_indices:
                    continue

                overlap = self._ngram_overlap(summary_sentence.ngrams, source_sentence.ngrams)
                if overlap > best_score:
                    best_score = overlap
                    best_sentence_idx = source_sentence_idx
                    if overlap >= 0.99:
                        break

            if best_sentence_idx:
                used_indices.add(best_sentence_idx)

            best_matches.append(
                MatchedSentence(
                    summary_sentence=summary_sentence.sentence,
                    best_sentence=source_sentence_dict[best_sentence_idx].sentence,
                    best_score=best_score,
                )
            )
        return best_matches

    def create_reference_summary(self, source: str, summary: str) -> str:
        """Creates a reference summary based on sentences from
        the source document that match the ones of the generated summary.
        """
        start_time = perf_counter()

        source_sentence_dict = self._get_sentence_data(source)
        summary_sentence_dict = self._get_sentence_data(summary)

        logger.debug(
            f"Source sentences {len(source_sentence_dict.keys())}\nSummary sentences {len(summary_sentence_dict.keys())}"
        )
        best_matches = self._find_matching_sentences(source_sentence_dict, summary_sentence_dict)
        reference_summary = " ".join(match.best_sentence for match in best_matches).strip()
        logger.info(f"Ngram extraction took {perf_counter() - start_time:.4f} seconds")
        return reference_summary
