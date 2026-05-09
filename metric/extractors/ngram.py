from collections import Counter
from time import perf_counter

from loguru import logger

from metric.extractors.base import Extractor
from metric.extractors.schemas import NgramSentenceData, MatchedSentence


class NgramExtractor(Extractor):
    """Extracts sentences from the source document that have n-gram overlap with the summary sentences."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def _get_sentence_data(self, text: str) -> dict[int, NgramSentenceData]:
        """Get sentences and ngrams from an input text.

        Args:
            text (str): Input candidate summary

        Returns:
            dict:
                - Sentence indices
                - Sentence extracted data
        """
        sentence_dict = {}

        # Parse the text through spaCy
        nlp = self.tokenizer(text.strip())
        for i, sent in enumerate(nlp.sents):
            # Skip the sentence if it's less than 5 words
            if len(sent) < 5:
                continue

            # Get the sentence lowercase tokens, after lemmatization, stopword and punctuation removal
            tokens = [token.lemma_.lower() for token in sent if not token.is_stop and not token.is_punct]

            # Token ngrams
            ngrams = Counter(tokens)
            sentence_dict[i] = NgramSentenceData(sentence=sent.text, ngrams=ngrams)
        return sentence_dict

    def _ngram_overlap(self, summary_ngrams: Counter, source_ngrams: Counter) -> float:
        """Compute ngram overlap between two sentences.

        Args:
            summary_ngrams (Counter): N-grams and their counts of the candidate summary
            source_ngrams (Counter): N-grams and their counts of the source document

        Returns:
            float: The overlap (ratio) of tokens between the two texts
        """
        total_summary_ngrams = sum(summary_ngrams.values())
        overlap = sum((summary_ngrams & source_ngrams).values())
        return overlap / total_summary_ngrams if total_summary_ngrams > 0 else 0

    def _find_matching_sentences(
        self, source_sentence_dict: dict, summary_sentence_dict: dict, overlap_threshold: float
    ) -> list[MatchedSentence]:
        """Find sentences with good overlap.

        Args:
            source_sentence_dict (dict): The source document data for all sentences.
            summary_sentence_dict (dict): The candidate summary data for all sentences.
            overlap_threshold (float): An overlap threshold to determine the best sentence early

        Returns:
            list[MatchedSentence]: A list with the best source document sentences for the candidate summary.
        """
        best_matches = []
        used_indices = set()

        for summary_sentence in summary_sentence_dict.values():
            best_score = -1
            best_sentence_idx = None

            for source_sentence_idx, source_sentence in source_sentence_dict.items():
                # Skip source sentences already picked up for the proxy reference summary
                if source_sentence_idx in used_indices:
                    continue

                overlap = self._ngram_overlap(summary_sentence.ngrams, source_sentence.ngrams)

                # Keep the sentence with the best overlap
                if overlap > best_score:
                    best_score = overlap
                    best_sentence_idx = source_sentence_idx

                    # unless it surpasses the predefined threshold
                    if overlap >= overlap_threshold:
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

    def extract_reference_summary(self, source: str, summary: str, overlap_threshold: float = 0.99) -> str:
        """Extracts a reference summary based on sentences from
        the source document that match the ones of the generated summary.

        Args:
            source (str): The original source document
            summary (str): The candidate summary to be evaluated
            overlap_threshold (float): An overlap threshold to determine the best sentence early

        Returns:
            str: A proxy reference summary based on the overlap between the two input texts
        """
        start_time = perf_counter()

        source_sentence_dict = self._get_sentence_data(source)
        summary_sentence_dict = self._get_sentence_data(summary)

        logger.debug(
            f"Source sentences {len(source_sentence_dict.keys())}\nSummary sentences {len(summary_sentence_dict.keys())}"
        )
        best_matches = self._find_matching_sentences(source_sentence_dict, summary_sentence_dict, overlap_threshold)
        reference_summary = " ".join(match.best_sentence for match in best_matches).strip()
        logger.info(f"Ngram extraction took {perf_counter() - start_time:.4f} seconds")
        return reference_summary, best_matches
