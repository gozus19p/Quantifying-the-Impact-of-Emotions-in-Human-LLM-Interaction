from typing import Any, Optional, Dict, List

from logic.experiment.model import Case

from pymongo import MongoClient
from pymongo.database import Database
from pymongo.collection import Collection
from logic.evaluation.model import Evaluation


class MongoDB:
    __client: MongoClient
    __database: Database
    __collection: Collection
    __CASE_COLLECTION: str = Case.__name__
    __EVALUATION_COLLECTION: str = Evaluation.__name__

    def __init__(self, mongodb_url: str, database: str):
        self.__client = MongoClient(mongodb_url)
        self.__database = self.__client[database]
        self.__collection = self.__database[self.__CASE_COLLECTION]
    
    def count_cases(self, case: Case) -> int:
        return self.__CASE_COLLECTION.count_documents(case.__dict__)
    
    def insert_many_cases(self, cases: List[Case]) -> None:
        assert all(c.valid_for_insert() for c in cases), "One or more cases is not valid for insertion"
        self.__CASE_COLLECTION.insert_many([c.__dict__ for c in cases])
    
    def get_last_record_identifier(self, case: Case) -> Optional[float]:
        result: Dict[str, Any] = self.__CASE_COLLECTION.find_one(case.__dict__, sort=[("record_identifier", -1)])
        return result["record_identifier"] if result else None
    
    def record_evaluation(self, evaluation: Evaluation) -> None:
        evaluation_dict: Dict[str, Any] = evaluation.__dict__()
        self.__EVALUATION_COLLECTION.insert_one(evaluation_dict)
