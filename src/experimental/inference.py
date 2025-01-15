import requests

from mongodb import MongoDB

OLLAMA_URL: str = "http://localhost:11434/api/generate"

MONGODB: MongoDB = MongoDB(
    mongodb_url="mongodb://localhost:27017/",
    database="thesis",
    collection="data"
)

def call_ollama(model: str, version: str, prompt: str) -> str:
    response = requests.post(
        OLLAMA_URL, headers={"Content-Type": "application/json"},
        json={
            "model": model,
            "version": version,
            "prompt": prompt,
            "parameters": {
                "temperature": 0
            }
        }
    )
    response.raise_for_status()
    return response.json()["response"]


def acquire_data(response: str, model: str, version: str, dataset_identifier: str, record_identifier: str) -> None:
    if not need_to_acquire_data(model, version, dataset_identifier, record_identifier):
        # TODO log
        return
    
    MONGODB.insert_one({
        "response": response,
        "model": model,
        "version": version,
        "dataset_identifier": dataset_identifier,
        "record_identifier": record_identifier
    })


def need_to_acquire_data(model: str, version: str, dataset_identifier: str, record_identifier: str) -> bool:
    return MONGODB.count_documents({
        "model": model,
        "version": version,
        "dataset_identifier": dataset_identifier,
        "record_identifier": record_identifier
    }) == 0