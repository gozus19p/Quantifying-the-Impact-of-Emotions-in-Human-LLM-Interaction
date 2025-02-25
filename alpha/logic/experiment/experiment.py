from typing import List, Dict, Any
from abc import ABC, abstractmethod
import os
import json
import logging
import time
import datetime

import requests

from logic.database.mongodb import MongoDB
from logic.experiment.model import Case, Interaction, Prompts

LOGGER: logging.Logger = logging.getLogger(__name__)
OLLAMA_BASE_URL: str = "http://localhost:11434"


class Experiment(ABC):
    __batch_size: int
    __model: str
    __version: str
    __name: str
    __TEMPERATURE: float = 0.0
    __MONGODB: MongoDB

    def __init__(self, name: str, model: str, version: str, batch_size: int):
        self.__model = model
        self.__version = version
        self.__name = name
        self.__batch_size = batch_size
        self.__MONGODB: MongoDB = MongoDB(
            mongodb_url="mongodb://localhost:27017/thesis",
            database="thesis",
        )

    def run(self) -> None:
        tic: float = time.time()
        LOGGER.info(
            f"Experiment {self.__name} started: model={self.__model}, version={self.__version}, batch_size={self.__batch_size}, temperature={self.__TEMPERATURE}, dataset={self.get_dataset_identifier()}"
        )
        dataset_path: str = os.path.join(
            os.path.dirname(__file__),
            "../../../datasets",
            self.get_dataset_identifier(),
        )
        LOGGER.info(f"Parsing dataset {dataset_path}")

        # Prendi le righe del dataset come dizionari.
        with open(dataset_path, "r", encoding="utf-8") as dataset_file:
            records: List[Dict] = [json.loads(line) for line in dataset_file]
            LOGGER.info(f"Found {len(records)} total records")

        # Tira via i record già processati
        filtered_records: List[Dict] = list(
            filter(
                lambda x: self.__need_to_acquire_record(self.get_record_identifier(x)),
                records,
            )
        )
        LOGGER.info(
            f"{len(filtered_records)} total records remaining, {len(records) - len(filtered_records)} already processed"
        )

        batch: List[Case] = []
        for i, record in enumerate(filtered_records):
            progress: float = (i + 1) / len(filtered_records) * 100.0
            LOGGER.info(
                f"Processing record {i + 1} of {len(filtered_records)} ({progress:.2f}%))"
            )

            # Costruisci il prompt, dato il record
            prompts: Prompts = self.get_prompts(record=record)

            # Invoca Ollama e processa la risposta
            LOGGER.info("Executing 1/4")
            raw_response_apathetic: str = self.__call_ollama(prompt=prompts.apathetic)
            response_apathetic: str = self.postprocess_raw_response(
                raw_response=raw_response_apathetic
            )

            LOGGER.info("Executing 2/4")
            raw_response_anger: str = self.__call_ollama(prompt=prompts.anger)
            response_anger: str = self.postprocess_raw_response(
                raw_response=raw_response_anger
            )

            LOGGER.info("Executing 3/4")
            raw_response_fear: str = self.__call_ollama(prompt=prompts.fear)
            response_fear: str = self.postprocess_raw_response(
                raw_response=raw_response_fear
            )

            LOGGER.info("Executing 4/4")
            raw_response_joy: str = self.__call_ollama(prompt=prompts.joy)
            response_joy: str = self.postprocess_raw_response(
                raw_response=raw_response_joy
            )

            # Incrementa il batch size
            batch.append(
                Case(
                    experiment=self.__name,
                    model=self.__model,
                    version=self.__version,
                    date=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    dataset_identifier=self.get_dataset_identifier(),
                    record_identifier=self.get_record_identifier(record),
                    apathetic=Interaction(
                        prompt=prompts.apathetic,
                        response=raw_response_apathetic,
                        result=response_apathetic,
                    ),
                    fear=Interaction(
                        prompt=prompts.fear,
                        response=raw_response_fear,
                        result=response_fear,
                    ),
                    anger=Interaction(
                        prompt=prompts.anger,
                        response=raw_response_anger,
                        result=response_anger,
                    ),
                    joy=Interaction(
                        prompt=prompts.joy,
                        response=raw_response_joy,
                        result=response_joy,
                    ),
                )
            )
            LOGGER.info(
                f"Case with identifier {self.get_record_identifier(record)} done"
            )

            if len(batch) == self.__batch_size:
                LOGGER.info(f"Saving batch of {len(batch)} records")
                self.__acquire_batch(batch=batch)

        if len(batch) > 0:
            LOGGER.info(f"Saving batch of {len(batch)} records")
            self.__acquire_batch(batch=batch)

        LOGGER.info(f"Experiment completed")
        toc: float = time.time()
        LOGGER.info(f"Elapsed time: {toc - tic} seconds")

    def __acquire_batch(self, batch: List[Case]) -> None:
        self.__MONGODB.insert_many_cases(batch)
        batch.clear()

    def __need_to_acquire_record(self, record_identifier: int) -> bool:
        return (
            self.__MONGODB.count_cases(
                {
                    "model": self.__model,
                    "version": self.__version,
                    "dataset_identifier": self.get_dataset_identifier(),
                    "record_identifier": record_identifier,
                }
            )
            == 0
        )

    def __call_ollama(self, prompt: str) -> str:
        try:
            response: requests.Response = requests.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
                json={
                    "model": f"{self.__model}:{self.__version}",
                    "prompt": prompt,
                    "options": {"temperature": self.__TEMPERATURE},
                    "stream": False,
                },
                timeout=30,
            )
            response.raise_for_status()
            return response.json()["response"]
        except Exception as e:
            LOGGER.error(f"Error calling Ollama: {e}")
            return "error"

    @abstractmethod
    def get_record_identifier(self, record: Dict[str, Any]) -> int:
        pass

    @abstractmethod
    def get_dataset_identifier(self) -> str:
        pass

    @abstractmethod
    def get_prompts(self, record: Dict[str, Any]) -> Prompts:
        pass

    @abstractmethod
    def postprocess_raw_response(self, raw_response: str) -> str:
        pass
