from dataclasses import dataclass


@dataclass
class FaithfulnessScore:
    score: float
    unfaithful_statements: list[int]


@dataclass
class ConcisenessScore:
    score: float
    redundant_statements: list[int]


@dataclass
class SumEvalScore:
    score: float
    faithfulness: FaithfulnessScore
    conciseness: ConcisenessScore
