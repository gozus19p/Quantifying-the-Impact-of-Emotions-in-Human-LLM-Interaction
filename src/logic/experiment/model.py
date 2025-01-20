from dataclasses import dataclass
from typing import Optional, Dict, List


@dataclass
class Prompts:
    apathetic: str
    fear: str
    anger: str
    joy: str


@dataclass
class Interaction:
    prompt: str
    response: str
    result: str


@dataclass
class Case:
    experiment: Optional[str] = None
    model: Optional[str] = None
    version: Optional[str] = None
    date: Optional[str] = None
    dataset_identifier: Optional[str] = None
    record_identifier: Optional[int] = None
    fear: Optional[Interaction] = None
    anger: Optional[Interaction] = None
    apathetic: Optional[Interaction] = None
    joy: Optional[Interaction] = None


    def valid_for_insert(self) -> bool:
        return all(v is not None for v in vars(self).values())
