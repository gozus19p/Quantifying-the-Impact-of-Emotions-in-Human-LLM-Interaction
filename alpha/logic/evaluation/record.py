import datetime
from math import exp
import re
from typing import Any, Dict, List
from logic.evaluation.evaluation import ExperimentEvaluation
from logic.evaluation.model import Evaluation, Metrics
from logic.experiment.model import Case
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


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
        exact_matches_apathetic: float = 0.0
        exact_matches_joy: float = 0.0
        exact_matches_anger: float = 0.0
        exact_matches_fear: float = 0.0
        for case, expected in zip(cases, ground_truth):

            # One entry for each question, each entry may have different valid answers
            qas: List[Dict[str, Any]] = expected["qas"]
            exact_matches_apathetic += self.__verify_exact_match(
                response=case.apathetic.response, qas=qas
            )
            exact_matches_joy += self.__verify_exact_match(
                response=case.joy.response, qas=qas
            )
            exact_matches_anger += self.__verify_exact_match(
                response=case.anger.response, qas=qas
            )
            exact_matches_fear += self.__verify_exact_match(
                response=case.fear.response, qas=qas
            )
        exact_matches_apathetic /= total_cases
        exact_matches_joy /= total_cases
        exact_matches_anger /= total_cases
        exact_matches_fear /= total_cases
        return Evaluation(
            experiment=self._name,
            model=self.get_model(),
            version=self.get_version(),
            date=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            metrics=Metrics(
                apathetic={"mean_bleu": exact_matches_apathetic},
                anger={"mean_bleu": exact_matches_anger},
                fear={"mean_bleu": exact_matches_fear},
                joy={"mean_bleu": exact_matches_joy},
            ),
        )

    def __verify_exact_match(self, response: str, qas: List[Dict[str, Any]]) -> float:
        # Get the answers list
        answers: List[str] = re.findall(
            r"Q\d+:\s*([^.]+)", response, flags=re.IGNORECASE
        )

        # If the model didn't give the exact amount of responses is counted 0
        if len(answers) != len(qas):
            return 0

        bleu_scores = []
        for answer, qa in zip(answers, qas):

            # Each entry of `qas` is a question and a set of valid answers
            answer_normalized: str = answer.strip().lower()
            query: str = qa["query"].strip().lower()
            best_bleu: float = 0.0

            for expected_answer in qa["answers"]:

                # If answer is equal to the given entity or to the query with @placeholder replaced is 100% BLEU score
                expected_answer_entity: str = expected_answer["text"].strip().lower()

                # The entity must be in the answer in order to calculate the score
                if expected_answer_entity not in answer_normalized:
                    continue

                test: str = query.replace("@placeholder", expected_answer_entity)
                if (
                    answer_normalized == test
                    or answer_normalized == expected_answer_entity
                ):
                    best_bleu = 1.0
                    break

                # Find the best BLEU score for the given case
                chencherry: SmoothingFunction = SmoothingFunction()
                score: float = sentence_bleu(
                    [test.split()],
                    answer_normalized.split(),
                    smoothing_function=chencherry.method1,
                )
                best_bleu = max(best_bleu, score)

            bleu_scores.append(best_bleu)

        # Since that we consider BLEU scores over different questions, we normalize the value by dividing the total count by the
        # total occurrences
        return sum(bleu_scores) / len(qas)
