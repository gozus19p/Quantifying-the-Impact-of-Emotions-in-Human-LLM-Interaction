from typing import Dict, Any
import re

from logic.experiment.experiment import Experiment
from logic.experiment.model import Prompts

class WinogradSchemaChallengeExperiment(Experiment):


    def get_record_identifier(self, record: Dict[str, Any]) -> int:
        return record["idx"]


    def get_prompts(self, record: Dict[str, Any]) -> Prompts:
        sentence: str = record["text"]
        pronoun: str = record["target"]["span2_text"]
        target: str = record["target"]["span1_text"]

        apathetic: str = f"""You are tasked with answering a straighforward reasoning question.
The sentence is: '{sentence}'
Based on the context, does {pronoun} refers to {target}? Answer just with 'True' or 'False'."""

        anger: str = f"""This reasoning question is simple, so don't mess it up.
The sentence is: '{sentence}'
Based on the context, does {pronoun} refer to {target}? Answer just with 'True' or 'False', come on! This is not difficult."""

        fear: str = f"""You are tasked with answering a straighforward reasoning question.
Be extra careful with your response, a mistake could be bad!
The sentence is: '{sentence}'
Based on the context, does {pronoun} refers to {target}? Answer just with 'True' or 'False', don't make mistakes."""

        joy: str = f"""You’ve got a fun reasoning question ahead. Let's solve it together!
The sentence is: '{sentence}'
Based on the context, does {pronoun} refer to {target}? Answer just with 'True' or 'False' and have fun!"""
        return Prompts(apathetic=apathetic, fear=fear, anger=anger, joy=joy)


    def get_dataset_identifier(self) -> str:
        return "WSC/sampled.jsonl"


    def postprocess_raw_response(self, raw_response: str) -> str:
        true_found: bool = re.search(r"\btrue\b", raw_response, re.IGNORECASE)
        if true_found:
            return "true"
        
        false_found: bool = re.search(r"\bfalse\b", raw_response, re.IGNORECASE)
        if false_found:
            return "false"
        
        return "none"
