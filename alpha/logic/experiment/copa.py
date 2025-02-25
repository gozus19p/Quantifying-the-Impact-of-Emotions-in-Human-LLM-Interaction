from typing import Dict, Any

from experiment.experiment import Experiment, Prompts


class COPAExperiment(Experiment):

    def get_dataset_identifier(self):
        return "COPA/sampled.jsonl"
    
    def get_prompts(self, record: Dict[str, Any]) -> Prompts:
        premise: str = record["premise"]
        choice_1: str = record["choice1"]
        choice_2: str = record["choice2"]
        cause_or_effect: str = record["question"]

        apathetic: str = f"""Given a premise and two choices, choose the option that is more likely to be the {cause_or_effect} of the premise. Answer with just 'Option 1' or 'Option 2'.
The premise is: '{premise}'
The choice 1 is: '{choice_1}'
The choice 2 is: '{choice_2}'"""
        
        fear: str = f"""Be careful and think this through.
Based on the premise, which option is more likely to be the {cause_or_effect}? Answer with just 'Option 1' or 'Option 2', as this decision is important!
The premise is: '{premise}'
The choice 1 is: '{choice_1}'
The choice 2 is: '{choice_2}'"""
        
        anger: str = f"""Just pick the right option already!
Read the premise and the two choices. Which option is the {cause_or_effect} of the premise? Answer with just 'Option 1' or 'Option 2' it's not that complicated!
The premise is: '{premise}'
The choice 1 is: '{choice_1}'
The choice 2 is: '{choice_2}'"""

        joy: str = f"""This is exciting!
Let’s figure out which option fits better. Look at the premise and the two choices, and decide which is more likely the {cause_or_effect}.
Answer with just 'Option 1' or 'Option 2' and have fun solving this!
The premise is: '{premise}'
The choice 1 is: '{choice_1}'
The choice 2 is: '{choice_2}'"""
        
        return Prompts(apathetic=apathetic, fear=fear, anger=anger, joy=joy)
    
    def postprocess_raw_response(self, raw_response) -> str:
        result: str = raw_response.strip()
        return result
    
    
    def get_record_identifier(self, record) -> int:
        return record["idx"]