import spacy

from metric.tokenizers.models_registry import MODELS


class Tokenizer:
    _LOADED_MODELS = {}

    def __init__(self, lang_code):
        models = MODELS

        if lang_code not in models:
            raise ValueError(f"No spaCy model available for language: {lang_code}")

        if lang_code not in Tokenizer._LOADED_MODELS:
            Tokenizer._LOADED_MODELS[lang_code] = spacy.load(models[lang_code])

        self.nlp = Tokenizer._LOADED_MODELS[lang_code]
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
