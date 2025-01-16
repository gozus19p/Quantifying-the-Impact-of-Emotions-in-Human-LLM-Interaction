import logging
from typing import List

from logic.experiment.boolean_qa import BooleanQAExperiment
from logic.evaluation.boolean_qa import BooleanQAExperimentEvaluation


logging.basicConfig(level=logging.INFO)


def main() -> None:
    models: List[str] = [
        "mistral:7b-instruct-q4_0",
        "llama3.1:8b-instruct-q4_0",
        "qwen2.5:7b-instruct-q4_0",
        "gemma2:9b-instruct-q4_0",
        "olmo2:7b-1124-instruct-q4_K_M"
    ]
    for model in models:
        model_name, model_version = model.split(":")
        experiment: BooleanQAExperiment = BooleanQAExperiment(model=model_name, version=model_version, batch_size=10)
        experiment.run()
        evaluation: BooleanQAExperimentEvaluation = BooleanQAExperimentEvaluation(model=model_name, version=model_version)
        evaluation.evaluate()
    

if __name__ == "__main__":
    main()