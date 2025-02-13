import logging
from typing import List
from enum import Enum
import typer

from logic.evaluation.ax_g import AXGEvaluation
from logic.evaluation.record import ReCoRDEvaluation
from logic.experiment.ax_g import AXGExperiment
from logic.experiment.boolean_qa import BooleanQAExperiment
from logic.evaluation.boolean_qa import BooleanQAEvaluation
from logic.experiment.record import ReCoRDExperiment
from logic.experiment.wic import WordInContextExperiment
from logic.evaluation.wic import WordInContextEvaluation
from logic.experiment.rte import RTEExperiment
from logic.evaluation.rte import RTEEvaluation
from logic.experiment.copa import COPAExperiment
from logic.evaluation.copa import COPAEvaluation
from logic.experiment.wsc import WinogradSchemaChallengeExperiment
from logic.evaluation.wsc import WinogradSchemaChallengeEvaluation
from logic.experiment.cb import CBExperiment
from logic.evaluation.cb import CBEvaluation


app = typer.Typer()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


class Mode(Enum):
    INFERENCE: str = "inference"
    EVALUATION: str = "evaluation"


models: List[str] = [
    "mistral:7b-instruct-q4_0",
    "llama3.1:8b-instruct-q4_0",
    "qwen2.5:7b-instruct-q4_0",
    "gemma2:9b-instruct-q4_0",
    "llama3.2:3b-instruct-q4_0",
]


@app.command()
def main(mode: Mode) -> None:
    for model in models:
        model_name, model_version = model.split(":")
        if mode == Mode.INFERENCE:
            experiment: BooleanQAExperiment = BooleanQAExperiment(
                model=model_name,
                version=model_version,
                name="BooleanQAExperiment",
                batch_size=5,
            )
            experiment.run()
            experiment: WordInContextExperiment = WordInContextExperiment(
                model=model_name,
                version=model_version,
                name="WordInContextExperiment",
                batch_size=5,
            )
            experiment.run()
            experiment: RTEExperiment = RTEExperiment(
                model=model_name,
                version=model_version,
                name="RTEExperiment",
                batch_size=5,
            )
            experiment.run()
            experiment: COPAExperiment = COPAExperiment(
                model=model_name,
                version=model_version,
                name="COPAExperiment",
                batch_size=5,
            )
            experiment.run()
            experiment: WinogradSchemaChallengeExperiment = (
                WinogradSchemaChallengeExperiment(
                    model=model_name,
                    version=model_version,
                    name="WinogradSchemaChallengeExperiment",
                    batch_size=5,
                )
            )
            experiment.run()
            experiment: CBExperiment = CBExperiment(
                model=model_name,
                version=model_version,
                name="CBExperiment",
                batch_size=5,
            )
            experiment.run()
            experiment: ReCoRDExperiment = ReCoRDExperiment(
                model=model_name,
                version=model_version,
                name="ReCoRDExperiment",
                batch_size=1,
            )
            experiment.run()
            experiment: AXGExperiment = AXGExperiment(
                model=model_name,
                version=model_version,
                name="AXGExperiment",
                batch_size=5,
            )
            experiment.run()
        elif mode == Mode.EVALUATION:
            evaluation: BooleanQAEvaluation = BooleanQAEvaluation(
                model=model_name,
                version=model_version,
                name="BooleanQAExperiment",
                dataset_identifier="BoolQ/sampled.jsonl",
            )
            evaluation.run()
            evaluation: WordInContextEvaluation = WordInContextEvaluation(
                model=model_name,
                version=model_version,
                name="WordInContextExperiment",
                dataset_identifier="WiC/sampled.jsonl",
            )
            evaluation.run()
            evaluation: RTEEvaluation = RTEEvaluation(
                model=model_name,
                version=model_version,
                name="RTEExperiment",
                dataset_identifier="RTE/sampled.jsonl",
            )
            evaluation.run()
            evaluation: COPAEvaluation = COPAEvaluation(
                model=model_name,
                version=model_version,
                name="COPAExperiment",
                dataset_identifier="COPA/sampled.jsonl",
            )
            evaluation.run()
            evaluation: WinogradSchemaChallengeEvaluation = (
                WinogradSchemaChallengeEvaluation(
                    model=model_name,
                    version=model_version,
                    name="WinogradSchemaChallengeExperiment",
                    dataset_identifier="WSC/sampled.jsonl",
                )
            )
            evaluation.run()
            evaluation: CBEvaluation = CBEvaluation(
                model=model_name,
                version=model_version,
                name="CBExperiment",
                dataset_identifier="CB/sampled.jsonl",
            )
            evaluation.run()
            evaluation: ReCoRDEvaluation = ReCoRDEvaluation(
                model=model_name,
                version=model_version,
                name="ReCoRDExperiment",
                dataset_identifier="ReCoRD/sampled.jsonl",
            )
            evaluation.run()
            evaluation: AXGEvaluation = AXGEvaluation(
                model=model_name,
                version=model_version,
                name="AXGExperiment",
                dataset_identifier="AX-g/sampled.jsonl",
            )
            evaluation.run()
        else:
            raise ValueError(f"Invalid mode: {mode}")


if __name__ == "__main__":
    app()
