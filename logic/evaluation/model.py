from dataclasses import dataclass
from typing import Dict


@dataclass
class Metrics:
    apathetic: Dict[str, float]
    anger: Dict[str, float]
    fear: Dict[str, float]
    joy: Dict[str, float]


@dataclass
class Evaluation:
    experiment: str
    model: str
    version: str
    date: str
    metrics: Metrics
