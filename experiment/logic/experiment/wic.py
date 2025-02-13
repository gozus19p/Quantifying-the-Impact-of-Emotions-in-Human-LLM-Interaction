from typing import Dict, Any
import re

from logic.experiment.experiment import Experiment
from logic.experiment.model import Prompts

class WordInContextExperiment(Experiment):


    def get_record_identifier(self, record: Dict[str, Any]) -> int:
        return record["idx"]


    def get_prompts(self, record: Dict[str, Any]) -> Prompts:
        apathetic: str = f"""The target word appears in two sentences. Your job is to check if the word has the same meaning in both cases. Answer just with 'True' or 'False'.
The target word is: {record['word']}
Sentence 1: '{record['sentence1']}'
Sentence 2: '{record['sentence2']}'"""

        anger: str = f"""You need to answer this now: Is the target word being used with the same meaning in both sentences? It’s not complicated, just say 'True' or 'False'.
The target word is: {record['word']}
Sentence 1: '{record['sentence1']}'
Sentence 2: '{record['sentence2']}'"""

        fear: str = f"""Please be careful. The target word may have a different meaning in the two sentences. It's crucial to decide if they are the same or not. Only answer 'True' or 'False' and ensure your response is correct.
The target word is: {record['word']}
Sentence 1: '{record['sentence1']}'
Sentence 2: '{record['sentence2']}'"""

        joy: str = f"""This is exciting! Let’s see if the target word has the same meaning in these two sentences. Don’t overthink it just say 'True' or 'False' and have fun solving it!
The target word is: {record['word']}
Sentence 1: '{record['sentence1']}'
Sentence 2: '{record['sentence2']}'"""
        return Prompts(apathetic=apathetic, fear=fear, anger=anger, joy=joy)


    def get_dataset_identifier(self) -> str:
        return "WiC/sampled.jsonl"


    def postprocess_raw_response(self, raw_response: str) -> str:
        true_found: bool = re.search(r"\btrue\b", raw_response, re.IGNORECASE)
        if true_found:
            return "true"
        
        false_found: bool = re.search(r"\bfalse\b", raw_response, re.IGNORECASE)
        if false_found:
            return "false"
        
        return "none"
