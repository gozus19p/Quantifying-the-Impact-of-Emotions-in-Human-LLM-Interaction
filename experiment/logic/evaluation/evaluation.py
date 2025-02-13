from abc import ABC, abstractmethod
from typing import List, Dict, Any
import logging
from os.path import dirname, join
import json

from logic.database.mongodb import MongoDB
from logic.experiment.model import Case
from logic.evaluation.model import Evaluation


LOGGER: logging.Logger = logging.getLogger(__name__)


class ExperimentEvaluation(ABC):
    __model: str
    __version: str
    _name: str
    __MONGODB: MongoDB

    def __init__(self, model: str, version: str, name: str):
        self.__MONGODB: MongoDB = MongoDB(
            mongodb_url="mongodb://localhost:27017/thesis",
            database="thesis",
        )
        self.__model = model
        self.__version = version
        self._name = name

    def get_model(self) -> str:
        return self.__model

    def get_version(self) -> str:
        return self.__version

    def run(self) -> None:
        LOGGER.info(
            f"Starting to evaluate experiment {self._name}: model={self.__model}, version={self.__version}"
        )
        total_evaluations: int = self.__MONGODB.count_evaluations(
            {"experiment": self._name, "model": self.__model, "version": self.__version}
        )
        if total_evaluations > 0:
            LOGGER.info("Experiment already evaluated. Skipping...")
            return

        cases: List[Case] = self.__MONGODB.find_many_cases(
            model=self.__model, version=self.__version, experiment=self._name
        )
        LOGGER.info(f"Found {len(cases)} cases to evaluate")
        ground_truth: List[Dict[str, Any]] = self.get_ground_truth()
        evaluation: Evaluation = self.compute_metrics(
            cases=cases, ground_truth=ground_truth
        )
        LOGGER.info("Recording evaluation")
        self.__MONGODB.record_evaluation(evaluation=evaluation)
        LOGGER.info("Done")

    def get_ground_truth(self) -> List[Dict[str, Any]]:

        with open(
            join(dirname(__file__), "../../../datasets", self.get_dataset_identifier()),
            "r",
            encoding="utf-8",
        ) as f:
            records: List[Dict] = [json.loads(line) for line in f]
        return records

    @abstractmethod
    def get_dataset_identifier(self) -> str:
        pass

    @abstractmethod
    def compute_metrics(
        self, cases: List[Case], ground_truth: List[Dict[str, Any]]
    ) -> Evaluation:
        pass
