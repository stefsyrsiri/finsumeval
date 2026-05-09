from abc import ABC, abstractmethod


class Tokenizer(ABC):
    @abstractmethod
    def tokenize(self, text: str) -> list[str]:
        pass


class Sentencizer(ABC):
    @abstractmethod
    def sentencize(self, text: str) -> list[str]:
        pass
