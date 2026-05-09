import spacy

from metric.tokenizers.base import Tokenizer, Sentencizer
from metric.tokenizers.models_registry import SPACY_MODELS


class SpacyTokenizer(Tokenizer, Sentencizer):
    def __init__(self, lang: str):
        self.lang = lang
        self._models = SPACY_MODELS
        self._loaded_models = {}

        if lang not in self._models:
            raise ValueError(f"No spaCy model available for language: {lang}")

        if lang not in self._loaded_models:
            self._loaded_models[lang] = spacy.load(self._models[lang])

        self.nlp = self._loaded_models[lang]
        if "sentencizer" not in self.nlp.pipe_names:
            self.nlp.add_pipe("sentencizer")
        self.nlp.max_length = 2_000_000

    def __call__(self, text):
        return self.nlp(text)

    def tokenize(self, nlp, clean=False):
        """Splits the text into tokens and optionally lemmatizes them, removes stopwords and punctuation."""
        if clean:
            return [token.lemma_ for token in nlp if not token.is_stop and not token.is_punct]
        else:
            return [token.text for token in nlp]

    def sentencize(self, nlp):
        """Splits the text into sentences."""
        return [sent.text for sent in nlp.sents]
