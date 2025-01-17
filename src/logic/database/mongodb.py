from typing import Any, Optional, Dict, List
from dataclasses import asdict

from logic.experiment.model import Case

from pymongo import MongoClient
from pymongo.database import Database
from pymongo.collection import Collection
from logic.evaluation.model import Evaluation


class MongoDB:
    __client: MongoClient
    __database: Database
    __case: Collection
    __evaluation: Collection

    def __init__(self, mongodb_url: str, database: str):
        self.__client = MongoClient(mongodb_url)
        self.__database = self.__client[database]
        self.__case = self.__database[Case.__name__]
        self.__evaluation = self.__database[Evaluation.__name__]
    
    def count_cases(self, case: Dict[str, Any]) -> int:
        return self.__case.count_documents(case)
    
    def insert_many_cases(self, cases: List[Case]) -> None:
        assert all(c.valid_for_insert() for c in cases), "One or more cases is not valid for insertion"
        self.__case.insert_many([asdict(c) for c in cases])
    
    def get_last_record_identifier(self, case: Case) -> Optional[float]:
        result: Dict[str, Any] = self.__case.find_one(asdict(case), sort=[("record_identifier", -1)])
        return result["record_identifier"] if result else None
    
    def record_evaluation(self, evaluation: Evaluation) -> None:
        evaluation_dict: Dict[str, Any] = asdict(evaluation)
        self.__evaluation.insert_one(evaluation_dict)
