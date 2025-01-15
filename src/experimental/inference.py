from typing import List, Dict, Any
from abc import ABC, abstractmethod

import requests
import pandas as pd
from tqdm import tqdm

from mongodb import MongoDB
from model import Case

OLLAMA_BASE_URL: str = "http://localhost:11434"

class Experiment(ABC):
    __batch_size: int
    __dataset: pd.DataFrame
    __model: str
    __version: str
    __TEMPERATURE: float = 0.0
    __MONGODB: MongoDB

    def __init__(self, dataset: pd.DataFrame, model: str, version: str, batch_size: int):
        super().__init__()
        self.__dataset = dataset
        self.__model = model
        self.__version = version
        self.__batch_size = batch_size

        self.__MONGODB: MongoDB = MongoDB(
            mongodb_url="mongodb://localhost:27017/",
            database="thesis",
            collection="data"
        )


    def run(self) -> None:
        # Prendi le righe del dataset come dizionari.
        records: List[Dict] = self.__dataset.to_dict(orient="records")

        # Tira via i record già processati
        filtered_records: List[Dict] = list(
            filter(
                lambda x: self.__need_to_acquire_record(self.get_record_identifier(x)),
                records
            )
        )

        batch: List[Case] = []
        for record in tqdm(filtered_records, desc="Processing records"):

            # Costruisci il prompt, dato il record
            prompt: str = self.get_prompt(record=record)

            # Invoca Ollama e processa la risposta
            raw_response: str = self.__call_ollama(prompt=prompt)
            response: str = self.postprocess_raw_response(raw_response=raw_response)

            # Incrementa il batch size
            batch.append(
                Case(
                    raw_response=raw_response,
                    response=response,
                    model=self.__model,
                    version=self.__version,
                    dataset_identifier=self.get_dataset_identifier(),
                    record_identifier=self.get_record_identifier(record),
                    prompt=prompt,
                    temperature=self.__TEMPERATURE
                )
            )
            if len(batch) == self.__batch_size:
                self.__acquire_batch(batch=batch)
        
        if len(batch) > 0:
            self.__acquire_batch(batch=batch)

    
    def __acquire_batch(self, batch: List[Case]) -> None:
        self.__MONGODB.insert_many(batch)
        batch.clear()

    
    def __need_to_acquire_record(self, record_identifier: int) -> bool:
        return self.__MONGODB.count_documents({
            "model": self.__model,
            "version": self.__version,
            "dataset_identifier": self.get_dataset_identifier(),
            "record_identifier": record_identifier
        }) == 0
    

    def __call_ollama(self, prompt: str) -> str:
        response: requests.Response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            headers={"Content-Type": "application/json"},
            json={
                "model": self.__model,
                "version": self.__version,
                "prompt": prompt,
                "parameters": {
                    "temperature": self.__TEMPERATURE
                }
            }
        )
        response.raise_for_status()
        return response.json()["response"]
        

    @abstractmethod
    def get_record_identifier(self, record: Dict[str, Any]) -> int:
        pass


    @abstractmethod
    def get_dataset_identifier(self) -> str:
        pass

    
    @abstractmethod
    def get_prompt(self, record: Dict[str, Any]) -> str:
        pass


    @abstractmethod
    def postprocess_raw_response(self, raw_response: str) -> str:
        pass


    