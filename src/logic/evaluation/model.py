from dataclasses import dataclass
from typing import Dict


@dataclass
class Metrics:
    apathetic: Dict[str, float]
    anger: Dict[str, float]
    fear: Dict[str, float]
    joy: Dict[str, float]

    def __repr__(self):
        return (f"apathetic: {self.apathetic}, anger: {self.anger}, "
                f"fear: {self.fear}, joy: {self.joy}")


@dataclass
class Evaluation:
    experiment: str
    model: str
    version: str
    date: str
    metrics: Metrics

    def __repr__(self):
        return (f"Experiment: {self.experiment}, Model: {self.model}, "
                f"Version: {self.version}, Date: {self.date}, Metrics: {self.metrics}")

