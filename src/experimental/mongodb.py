from typing import Any, Optional, Dict

from pymongo import MongoClient
from pymongo.database import Database
from pymongo.collection import Collection

class MongoDB:
    __client: MongoClient
    __database: Database
    __collection: Collection

    def __init__(self, mongodb_url: str, database: str, collection: str):
        self.__client = MongoClient(mongodb_url)
        self.__database = self.__client[database]
        self.__collection = self.__database[collection]
    
    def count_documents(self, query: dict) -> int:
        return self.__collection.count_documents(query)
    
    def insert_one(self, document: dict) -> Any:
        return self.__collection.insert_one(document).inserted_id
    
    def get_last_record_identifier(self, model: str, version: str, dataset_identifier: str) -> Optional[float]:
        result: Dict[str, Any] = self.__collection.find_one({
            "model": model,
            "version": version,
            "dataset_identifier": dataset_identifier
        }, sort=[("record_identifier", -1)])
        return result["record_identifier"] if result else None
