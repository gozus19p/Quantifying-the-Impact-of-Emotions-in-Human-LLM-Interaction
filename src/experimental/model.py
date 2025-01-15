from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class Case:
    response: Optional[str] = None
    model: Optional[str] = None
    version: Optional[str] = None
    dataset_identifier: Optional[str] = None
    record_identifier: Optional[int] = None
    prompt: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None

    def valid_for_insert(self) -> bool:
        return all(vars(self).values())