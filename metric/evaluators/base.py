from abc import ABC, abstractmethod

from metric.evaluators.schemas import ConcisenessScore, FaithfulnessScore, SumEvalScore


class Evaluator(ABC):
    @abstractmethod
    def score(self, summary: str, source: str) -> SumEvalScore:
        """Calculate the overall score of a summary."""
        pass

    @abstractmethod
    def score_faithfulness(self, summary: str, source: str) -> FaithfulnessScore:
        """Calculate the faithfulness score of a summary."""
        pass

    @abstractmethod
    def score_conciseness(self, summary: str) -> ConcisenessScore:
        """Calculate the conciseness score of a summary."""
        pass
