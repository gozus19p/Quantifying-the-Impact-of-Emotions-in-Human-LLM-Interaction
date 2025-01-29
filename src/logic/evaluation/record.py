from math import exp
import re
from typing import Any, Dict, List
from logic.evaluation.evaluation import ExperimentEvaluation
from logic.evaluation.model import Evaluation
from logic.experiment.model import Case


class ReCoRDEvaluation(ExperimentEvaluation):

    __dataset_identifier: str

    def __init__(self, model: str, version: str, name: str, dataset_identifier: str):
        super().__init__(model=model, version=version, name=name)
        self.__dataset_identifier = dataset_identifier

    def get_dataset_identifier(self) -> str:
        return self.__dataset_identifier

    def compute_metrics(
        self, cases: List[Case], ground_truth: List[Dict[str, Any]]
    ) -> Evaluation:

        # Sorting to ensure each pair matches by `idx`
        cases.sort(key=lambda c: c.record_identifier)
        ground_truth.sort(key=lambda g: g["idx"])
        assert all(c.record_identifier == g["idx"] for c, g in zip(cases, ground_truth))

        total_cases: int = len(cases)
        exact_matches_apathetic: int = 0
        exact_matches_joy: int = 0
        exact_matches_anger: int = 0
        exact_matches_fear: int = 0
        for case, expected in zip(cases, ground_truth):

            # One entry for each question, each entry may have different valid answers
            qas: List[Dict[str, Any]] = [qas for qas in expected["qas"]]
            pass

        return super().compute_metrics(cases, ground_truth)

    def __verify_exact_match(self, response: str, expected: List[str]) -> int:
        # Models do not follow this format, consider broader rules
        answers: List[str] = re.findall(
            r"Q\d+:\s*([^.]+)", response, flags=re.IGNORECASE
        )
        if len(answers) != len(expected):
            return 0

        for answer, expected_answer in zip(answers, expected):
            if answer.strip().lower() != expected_answer.strip().lower():
                return 0

        return 1
