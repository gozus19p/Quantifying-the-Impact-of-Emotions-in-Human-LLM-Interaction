import logging
from typing import List
from enum import Enum

from logic.experiment.boolean_qa import BooleanQAExperiment
from logic.evaluation.boolean_qa import BooleanQAExperimentEvaluation


logging.basicConfig(level=logging.INFO)

class Mode(Enum):
    INFERENCE = 0
    EVALUATION = 1


def main(mode: Mode) -> None:
    models: List[str] = [
        "mistral:7b-instruct-q4_0",
        "llama3.1:8b-instruct-q4_0",
        "qwen2.5:7b-instruct-q4_0",
        "gemma2:9b-instruct-q4_0",
        "llama3.2:3b-instruct-q4_0"
    ]
    for model in models:
        model_name, model_version = model.split(":")
        if mode == Mode.INFERENCE:
            experiment: BooleanQAExperiment = BooleanQAExperiment(
                model=model_name, version=model_version, name="BooleanQAExperiment", batch_size=5
            )
            experiment.run()
        
        if mode == Mode.EVALUATION:
            evaluation: BooleanQAExperimentEvaluation = BooleanQAExperimentEvaluation(
                model=model_name, version=model_version, name="BooleanQAExperiment"
            )
            evaluation.run()
    

if __name__ == "__main__":
    main(Mode.EVALUATION)