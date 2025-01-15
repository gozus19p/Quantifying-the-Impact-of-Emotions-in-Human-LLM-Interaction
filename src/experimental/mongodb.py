from typing import Any, Optional, Dict

from model import Case

from pymongo import MongoClient
from pymongo.database import Database
from pymongo.collection import Collection

class MongoDB:
    __client: MongoClient
    __database: Database
    __collection: Collection
    __COLLECTION_NAME: str = Case.__name__

    def __init__(self, mongodb_url: str, database: str):
        self.__client = MongoClient(mongodb_url)
        self.__database = self.__client[database]
        self.__collection = self.__database[self.__COLLECTION_NAME]
    
    def count_documents(self, case: Case) -> int:
        return self.__collection.count_documents(case.__dict__)
    
    def insert_one(self, case: Case) -> Any:
        assert case.valid_for_insert(), "Case is not valid for insertion"
        return self.__collection.insert_one(case.__dict__) \
            .inserted_id
    
    def get_last_record_identifier(self, case: Case) -> Optional[float]:
        result: Dict[str, Any] = self.__collection.find_one(case.__dict__, sort=[("record_identifier", -1)])
        return result["record_identifier"] if result else None
