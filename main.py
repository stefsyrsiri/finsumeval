import os
from time import perf_counter

from dotenv import load_dotenv
from loguru import logger
from transformers import logging

from metric.extractors.ngram import NgramExtractor
from metric.evaluators.sumeval import SumEval
from metric.tokenizers.tokenizer import SpacyTokenizer

logging.set_verbosity_error()
load_dotenv()

NLI_MODEL = os.getenv("NLI_MODEL", "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli")
SOURCE_DOC = os.getenv("SOURCE_DOC", "data/17.txt")
SUMMARY = os.getenv("SUMMARY", "data/17_1.txt")


def create_proxy_ref(source, summary):
    logger.info("Starting n-gram extraction.")
    tokenizer = SpacyTokenizer("en")
    extractor = NgramExtractor(tokenizer)

    ref, best_matches = extractor.extract_reference_summary(source, summary)
    with open("data/17_extracted.txt", "w", encoding="utf-8") as f:
        f.write(ref)
    return ref


def finsumeval(source, summary, model):
    logger.info(f"Starting evaluation with {model}.")
    metric = SumEval(nli_model=model)

    logger.info("Scoring...")
    start_time = perf_counter()
    logger.info(metric.score(summary, source))
    logger.info(f"Time taken: {perf_counter() - start_time:.2f} seconds")


@logger.catch
def main():
    logger.info(f"Loading source document ({SOURCE_DOC}) and summary ({SUMMARY}).")
    with open(SOURCE_DOC, "r", encoding="utf-8") as f:
        source = f.read()
    with open(SUMMARY, "r", encoding="utf-8") as f:
        summary = f.read()
    ref = create_proxy_ref(source, summary)
    finsumeval(source=ref, summary=summary, model=NLI_MODEL)


if __name__ == "__main__":
    main()
