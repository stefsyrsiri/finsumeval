from abc import ABC, abstractmethod


class Extractor(ABC):
    @abstractmethod
    def extract_reference_summary(self, source: str, summary: str) -> str:
        """Extract a proxy reference summary from the source and candidate summary."""
        pass
