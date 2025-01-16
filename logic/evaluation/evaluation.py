from abc import ABC, abstractmethod
from typing import List, Dict, Any
import datetime
import logging

from logic.database.mongodb import MongoDB
from logic.experiment.model import Case
from logic.evaluation.model import Evaluation


LOGGER: logging.Logger = logging.getLogger(__name__)

class ExperimentEvaluation(ABC):
    __model: str
    __version: str
    __MONGODB: MongoDB

    def __init__(self, model: str, version: str):
        self.__MONGODB: MongoDB = MongoDB(
            mongodb_url="mongodb://localhost:27017/",
            database="thesis",
            collection="data"
        )
        self.__model = model
        self.__version = version
    
    def evaluate(self) -> None:
        LOGGER.info(f"Starting to evaluate experiment {self.experiment}: model={self.__model}, version={self.__version}")
        cases: List[Case] = []  # TODO
        evaluation: Evaluation = self.compute_metrics(cases=cases)
        LOGGER.info("Recording evaluation")
        self.__MONGODB.record_evaluation(evaluation=evaluation)
        LOGGER.info("Done")


    @abstractmethod
    def compute_metrics(self, cases: List[Case], ground_truth: List[Dict[str, Any]]) -> Evaluation:
        pass