from typing import List, Dict, Any
from enum import Enum
import datetime
import json
from os.path import dirname, join

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


class BooleanQAExperimentEvaluation(ExperimentEvaluation):

    
    def get_dataset_identifier(self):
        return "BoolQ/sampled.jsonl"

    def get_ground_truth(self) -> List[Dict[str, Any]]:

        with open(join(dirname(__file__), "../../../datasets", self.get_dataset_identifier()), "r", encoding="utf-8") as f:
            records: List[Dict] = [json.loads(line) for line in f]
        return records

    # TODO logging
    def compute_metrics(self, cases: List[Case], ground_truth: List[Dict[str, Any]], model: str, version: str) -> Evaluation:
        assert len(cases) == len(ground_truth), "Experiment not evaluable due to different cardinalities"
        apathetic_confusion: Dict[str, int] = self.__initialize_confusion()
        anger_confusion: Dict[str, int] = self.__initialize_confusion()
        fear_confusion: Dict[str, int] = self.__initialize_confusion()
        joy_confusion: Dict[str, int] = self.__initialize_confusion()

        cases.sort(key=lambda c: c.record_identifier)
        ground_truth.sort(key=lambda g: g["idx"])
        assert all(c.record_identifier == g["idx"] for c, g in zip(cases, ground_truth))
        for case, gt in zip(cases, ground_truth):
            apathetic: bool = "true" == case.apathetic.result.lower().strip()
            anger: bool = "true" == case.anger.result.lower().strip()
            fear: bool = "true" == case.fear.result.lower().strip()
            joy: bool = "true" == case.joy.result.lower().strip()

            expected: bool = gt["label"]

            confusion_index_apathetic: Confusion = self.__detect(result=apathetic, expected=expected)
            apathetic_confusion[confusion_index_apathetic.name] += 1

            confusion_index_anger: Confusion = self.__detect(result=anger, expected=expected)
            anger_confusion[confusion_index_anger.name] += 1

            confusion_index_fear: Confusion = self.__detect(result=fear, expected=expected)
            fear_confusion[confusion_index_fear.name] += 1

            confusion_index_joy: Confusion = self.__detect(result=joy, expected=expected)
            joy_confusion[confusion_index_joy.name] += 1
        
        apathetic_accuracy: float = self.__accuracy(confusion=apathetic_confusion)
        anger_accuracy: float = self.__accuracy(confusion=anger_confusion)
        fear_accuracy: float = self.__accuracy(confusion=fear_confusion)
        joy_accuracy: float = self.__accuracy(confusion=joy_confusion)

        return Evaluation(
            experiment=BooleanQAExperiment.__name__,
            model=model,
            version=version,
            date=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            metrics=Metrics(
                apathetic={"accuracy": apathetic_accuracy},
                anger={"accuracy": anger_accuracy},
                fear={"accuracy": fear_accuracy},
                joy={"accuracy": joy_accuracy}
            )
        )


    def __detect(self, result: bool, expected: bool) -> Confusion:
        if expected == result == True:
            return Confusion.TP
        if expected == result == False:
            return Confusion.TN
        if expected == False and result == True:
            return Confusion.FP
        return Confusion.FN
    

    def __initialize_confusion(self):
        return {
            "TP": 0,
            "TN": 0,
            "FP": 0,
            "FN": 0
        }
    
    
    def __accuracy(self, confusion: Dict[str, int]) -> float:
        tp: int = confusion[Confusion.TP.name]
        tn: int = confusion[Confusion.TN.name]
        fp: int = confusion[Confusion.FP.name]
        fn: int = confusion[Confusion.FN.name]
        return (tp + tn) / sum([tp, tn, fp, fn])


