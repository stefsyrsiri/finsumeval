import os
from time import perf_counter

import pandas as pd
from bert_score import BERTScorer
from dotenv import load_dotenv
from loguru import logger
from transformers import logging

from metric.extractors.ngram import NgramExtractor
from metric.evaluators.ldfacts import LongDocFACTScore
from metric.evaluators.sumeval import SumEval
from metric.tokenizers.tokenizer import Tokenizer

logging.set_verbosity_error()
load_dotenv()

NLI_MODEL = os.getenv("NLI_MODEL", "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli")
SOURCE_DOC = os.getenv("SOURCE_DOC", "data/17.txt")
SUMMARY = os.getenv("SUMMARY", "data/17_1.txt")


def finsumeval(
    source,
    summary,
    model="joeddav/xlm-roberta-large-xnli",
):
    logger.info(f"Starting evaluation with {model}.")
    metric = SumEval(nli_model=model)

    logger.info("Scoring...")
    start_time = perf_counter()
    logger.info(metric.score(summary, source))
    logger.info(f"Time taken: {perf_counter() - start_time:.2f} seconds")


def ngram_extract(source, summary):
    logger.info("Starting n-gram extraction.")
    tokenizer = Tokenizer("en")
    extractor = NgramExtractor(tokenizer)

    ref = extractor.create_reference_summary(source, summary)
    # print(pd.DataFrame(ref).head())
    # pd.DataFrame(ref).to_json("ngram_extraction_results.jsonl", orient="records", lines=True)
    # with open("data/17_extracted.txt", "w", encoding="utf-8") as f:
    #     f.write(ref)
    return ref


def compute_ldfacts(source, summary, metric):
    ldfs = LongDocFACTScore(metric=metric)
    start_time = perf_counter()
    scores = ldfs.score_src_hyp_long(srcs=[source], hyps=[summary])
    logger.info(f"LongDocFACTScore: {scores}: Time taken: {perf_counter() - start_time:.2f} seconds")


def ngram_bertscore(source, summary):
    bertscorer = BERTScorer(model_type="ProsusAI/finbert")
    start_time = perf_counter()
    score = bertscorer.score(refs=[source], cands=[summary])
    logger.debug(f"Ref length {len(source)} - Cand length {len(summary)}")
    logger.info(f"Ngram BERTScore: {score}: Time taken: {perf_counter() - start_time:.2f} seconds")


@logger.catch
def main():
    logger.info(f"Loading source document ({SOURCE_DOC}) and summary ({SUMMARY}).")
    with open(SOURCE_DOC, "r", encoding="utf-8") as f:
        source = f.read()
    with open(SUMMARY, "r", encoding="utf-8") as f:
        summary = f.read()
    ref = ngram_extract(source, summary)
    finsumeval(source=ref, summary=summary, model=NLI_MODEL)
    # compute_ldfacts(source, summary, "BERTScore")
    # ngram_bertscore(ref, summary)


if __name__ == "__main__":
    main()
