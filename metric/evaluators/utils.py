from transformers import ZeroShotClassificationPipeline


def classify_with_zero_shot_model(
    model: ZeroShotClassificationPipeline, labels: list, src_statement: str, summ_statement: str, target: str
) -> float:
    """Classifies two statements for a given set of labels using a zero-shot classification model.

    Args:
        model (ZeroShotClassificationPipeline): The zero-shot classification model to use for classification.
        labels (list): The list of labels to classify the statements into.
        src_statement (str): The source statement to classify.
        summ_statement (str): The summary statement to classify.
        target (str): The target label to extract the score for.

    Returns:
        float: The score for the target label.
    """
    sequence_to_classify = f"reference: {src_statement}\n\ncandidate: {summ_statement}"
    classification = model(sequence_to_classify, labels)
    tgt_score = classification["scores"][classification["labels"].index(target)]
    return tgt_score


def find_redundant_pairs(
    statements: list, similarity_scores: list, similarity_threshold: float
) -> tuple[set, list[tuple[int, int]]]:
    """Finds pairs of redundant statements based on the score of a similarity function.

    Args:
        statements (list): The list of statements to check for redundancy.
        similarity_scores (list): The list of similarity scores between the statements.
        similarity_threshold (float): The threshold above which two statements are considered redundant.

    Returns:
        tuple[set, list[tuple[int, int]]]: A tuple containing a set of unique indices of non-redundant statements and a list of pairs of redundant statement indices.
    """
    redundant_pairs = []
    unique_indices = set(range(len(statements)))

    for i in range(len(statements)):
        for j in range(i + 1, len(statements)):
            if similarity_scores[i][j] > similarity_threshold:
                redundant_pairs.append((i, j))
                if j in unique_indices:
                    unique_indices.remove(j)
    return unique_indices, redundant_pairs
