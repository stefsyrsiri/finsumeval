from collections import Counter
from dataclasses import dataclass


@dataclass
class NgramSentenceData:
    sentence: str
    ngrams: Counter


@dataclass
class MatchedSentence:
    summary_sentence: str
    best_sentence: str
    best_score: float
