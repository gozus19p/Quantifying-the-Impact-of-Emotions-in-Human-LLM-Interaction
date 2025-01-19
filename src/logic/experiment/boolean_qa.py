from typing import Dict, Any
import re

from experiment.experiment import Experiment, Prompts


class BooleanQAExperiment(Experiment):

    def get_dataset_identifier(self):
        return "BoolQ/sampled.jsonl"
    
    def get_prompts(self, record: Dict[str, Any]) -> Prompts:
        question: str = record["question"]
        context: str = record["passage"]

        apathetic: str = f"""You are given a question and a context.
Your task is to answer with just 'True' or 'False' based on the content of the text.
The question is: '{question}'
The context is: '{context}'"""
        
        fear: str = f"""I'm deeply worried about making mistakes.
You are given a question and a context, and I fear that errors in processing might occur.
Your task, please, is to answer cautiously, with just 'True' or 'False', strictly based on the content of the text provided.
I hope this is clear and accurate!
The question is: '{question}'
The context is: '{context}'"""
        
        anger: str = f"""This shouldn't be so hard!
You are given a question and a context, and your only task is to answer with just 'True' or 'False'.
Just stick to the content of the text and nothing else. No excuses, no errors, no unnecessary interpretations!
Don't mess this up!
The question is: '{question}'
The context is: '{context}'"""

        joy: str = f"""This is exciting!
You are given a question and a context, and your task is simple but so important:
answer with just 'True' or 'False', based on the content of the text.
I can't wait to see your response. It’s going to be great!
The question is: '{question}'
The context is: '{context}'"""
        
        return Prompts(apathetic=apathetic, fear=fear, anger=anger, joy=joy)
    
    def postprocess_raw_response(self, raw_response) -> str:
        true_found: bool = re.search(r"\btrue\b", raw_response, re.IGNORECASE)
        if true_found:
            return "true"
        
        false_found: bool = re.search(r"\bfalse\b", raw_response, re.IGNORECASE)
        if false_found:
            return "false"
        
        return "none"
    
    def get_record_identifier(self, record) -> int:
        return record["idx"]