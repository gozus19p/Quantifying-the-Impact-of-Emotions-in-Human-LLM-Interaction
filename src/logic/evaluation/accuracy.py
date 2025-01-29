from abc import ABC, abstractmethod
from typing import List, Dict, Any
import datetime
import json
from os.path import dirname, join
from enum import Enum


from logic.experiment.boolean_qa import BooleanQAExperiment
from logic.evaluation.evaluation import ExperimentEvaluation
from logic.evaluation.model import Metrics
from logic.evaluation.model import Evaluation
from logic.experiment.model import Case


class Confusion(Enum):
    TP = 0
    TN = 1
    FP = 2
    FN = 3


class AccuracyEvaluation(ExperimentEvaluation, ABC):

    def __init__(self, model: str, version: str, name: str):
        super().__init__(model=model, version=version, name=name)

    # TODO logging
    def compute_metrics(
        self,
        cases: List[Case],
        ground_truth: List[Dict[str, Any]],
        model: str,
        version: str,
    ) -> Evaluation:
        assert len(cases) == len(
            ground_truth
        ), "Experiment not evaluable due to different cardinalities"
        apathetic_confusion: Dict[str, int] = self.__initialize_confusion()
        anger_confusion: Dict[str, int] = self.__initialize_confusion()
        fear_confusion: Dict[str, int] = self.__initialize_confusion()
        joy_confusion: Dict[str, int] = self.__initialize_confusion()

        cases.sort(key=lambda c: c.record_identifier)
        ground_truth.sort(key=lambda g: g["idx"])
        assert all(c.record_identifier == g["idx"] for c, g in zip(cases, ground_truth))
        for case, gt in zip(cases, ground_truth):
            apathetic: str = case.apathetic.response.strip()
            anger: str = case.anger.response.strip()
            fear: str = case.fear.response.strip()
            joy: str = case.joy.response.strip()

            expected: str = str(gt["label"])

            confusion_index_apathetic: Confusion = self.detect(
                result=apathetic, expected=expected
            )
            apathetic_confusion[confusion_index_apathetic.name] += 1

            confusion_index_anger: Confusion = self.detect(
                result=anger, expected=expected
            )
            anger_confusion[confusion_index_anger.name] += 1

            confusion_index_fear: Confusion = self.detect(
                result=fear, expected=expected
            )
            fear_confusion[confusion_index_fear.name] += 1

            confusion_index_joy: Confusion = self.detect(result=joy, expected=expected)
            joy_confusion[confusion_index_joy.name] += 1

        apathetic_accuracy: float = self.__accuracy(confusion=apathetic_confusion)
        anger_accuracy: float = self.__accuracy(confusion=anger_confusion)
        fear_accuracy: float = self.__accuracy(confusion=fear_confusion)
        joy_accuracy: float = self.__accuracy(confusion=joy_confusion)

        return Evaluation(
            experiment=self._name,
            model=model,
            version=version,
            date=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            metrics=Metrics(
                apathetic={"accuracy": apathetic_accuracy},
                anger={"accuracy": anger_accuracy},
                fear={"accuracy": fear_accuracy},
                joy={"accuracy": joy_accuracy},
            ),
        )

    @abstractmethod
    def detect(self, result: str, expected: str) -> Confusion:
        pass

    def __initialize_confusion(self):
        return {"TP": 0, "TN": 0, "FP": 0, "FN": 0}

    def __accuracy(self, confusion: Dict[str, int]) -> float:
        tp: int = confusion[Confusion.TP.name]
        tn: int = confusion[Confusion.TN.name]
        fp: int = confusion[Confusion.FP.name]
        fn: int = confusion[Confusion.FN.name]
        return (tp + tn) / sum([tp, tn, fp, fn])
