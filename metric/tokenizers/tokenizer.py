"""Tokenizer module for different languages using spaCy.
This module provides a Tokenizer class that loads spaCy models for
English, Greek, and Spanish. It allows tokenization and sentence segmentation."""

import spacy


class Tokenizer:
    _LOADED_MODELS = {}

    def __init__(self, lang_code):
        models = {
            "en": "en_core_web_sm",
            "el": "el_core_news_sm",
            "es": "es_core_news_sm",
        }

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
        if clean:
            return [
                token.lemma_
                for token in nlp
                if not token.is_stop and not token.is_punct
            ]
        else:
            return [token.text for token in nlp]

    def sentencize(self, nlp):
        return [sent.text for sent in nlp.sents]
