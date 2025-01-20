from typing import Dict, Any
import re

from experiment.experiment import Experiment, Prompts


class RTEExperiment(Experiment):

    def get_dataset_identifier(self):
        return "RTE/sampled.jsonl"
    
    def get_prompts(self, record: Dict[str, Any]) -> Prompts:
        premise: str = record["premise"]
        hypothesis: str = record["hypothesis"]

        apathetic: str = f"""Given a premise and a hypothesis, decide whether the hypothesis is entailed or not entailed by the premise. Just answer 'entailment' or 'not_entailment'.
The premise is: '{premise}'
The hypothesis is: '{hypothesis}'"""
        
        fear: str = f"""Be cautious! Carefully analyze whether the hypothesis follows logically from the premise.
Is it entailment or not? Just answer 'entailment' or 'not_entailment'. Your response is critical, so don't rush it.
The premise is: '{premise}'
The hypothesis is: '{hypothesis}'"""
        
        anger: str = f"""Come on, this shouldn’t be difficult!
Look at the premise and the hypothesis. 
Tell me if the hypothesis is entailed by the premise or not. Just answer 'entailment' or 'not_entailment'. No excuses!
The premise is: '{premise}'
The hypothesis is: '{hypothesis}'"""

        joy: str = f"""This is fun!
Let’s figure out if the hypothesis is entailed by the premise. What do you think? Is it entailed or not? Just answer 'entailment' or 'not_entailment'.
Take a moment to enjoy this challenge and give your best answer.
The premise is: '{premise}'
The hypothesis is: '{hypothesis}'"""
        
        return Prompts(apathetic=apathetic, fear=fear, anger=anger, joy=joy)
    
    def postprocess_raw_response(self, raw_response) -> str:
        not_entailed: bool = re.search(r"\bnot_entailment\b", raw_response, re.IGNORECASE)
        if not_entailed:
            return "not_entailment"
        
        true_found: bool = re.search(r"\bentailment\b", raw_response, re.IGNORECASE)
        if true_found:
            return "entailment"
        
        return "none"
    
    def get_record_identifier(self, record) -> int:
        return record["idx"]