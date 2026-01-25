import os
import time

from dotenv import load_dotenv
from loguru import logger
from transformers import logging

from metric.sumeval import SumEval

logging.set_verbosity_error()
load_dotenv()

NLI_MODEL = os.getenv("NLI_MODEL", "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli")
SOURCE_DOC = os.getenv("SOURCE_DOC", "data/17.txt")
SUMMARY = os.getenv("SUMMARY", "data/17_1.txt")

@logger.catch
def main(model):
    logger.info(f"Starting evaluation with {model}.")
    metric = SumEval(nli_model=model)

    logger.info(f"Loading source document ({SOURCE_DOC}) and summary ({SUMMARY}).")
    with open(SOURCE_DOC, "r", encoding="utf-8") as f:
        source = f.read()
    with open(SUMMARY, "r", encoding="utf-8") as f:
        summary = f.read()

    logger.info("Scoring...")
    start_time = time.time()
    logger.info(metric.score(summary, source))
    end_time = time.time()
    logger.info(f"Time taken: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main(model=NLI_MODEL)