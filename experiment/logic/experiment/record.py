from typing import Any, Dict, List
from logic.experiment.experiment import Experiment
from logic.experiment.model import Prompts


class ReCoRDExperiment(Experiment):

    def get_dataset_identifier(self):
        return "ReCoRD/sampled.jsonl"

    def get_prompts(self, record: Dict[str, Any]) -> Prompts:
        passage: str = record["passage"]["text"]
        questions: List[str] = [q["query"] for q in record["qas"]]

        # Merging questions together with format `Q{x}: {question}\n`
        questions_text: str = ""
        for i, question in enumerate(questions):
            questions_text += f"Q{i + 1}: {question}\n"

        questions_text = questions_text.strip()

        apathetic: str = f"""You are given a context and a set of questions.
Each question contains a placeholder.
Your task is to replace each placeholder with the correct word from the context.
Answer concisely using only the word that you would use to replace '@placeholder'.

Correct Format Example:
Q1: Madrid. Q2: Spain. Q3: Championship.

Incorrect Format Example (DO NOT DO THIS):
Q1: The city of Madrid is beautiful.

Context: '{passage}'
Questions:
{questions_text}"""

        joy: str = f"""Great! Let's solve these together!
You'll receive a context and some questions with missing words.
Find the correct word for each placeholder based on the given context.
Simply respond with the correct word that you would use to replace '@placeholder'.

Correct Format Example:
Q1: Madrid. Q2: Spain. Q3: Championship.

Incorrect Format Example (DO NOT DO THIS):
Q1: The city of Madrid is beautiful.

Context: '{passage}'
Questions:
{questions_text}"""

        angry: str = f"""Pay attention!
You are given a context and a set of questions with missing words.
Your job is to find the exact missing word that you would use to replace each '@placeholder'. Nothing more, nothing less.

Correct Format Example:
Q1: Madrid. Q2: Spain. Q3: Championship.

Incorrect Format Example (DO NOT DO THIS):
Q1: The city of Madrid is beautiful.

Context: '{passage}'
Questions:
{questions_text}"""

        fear: str = f"""Be careful! Accuracy is everything.
You are given a context and several questions, each missing a word.
Your task is to find only the missing word that you would use to replace each '@placeholder' in the context without any mistakes.

Correct Format Example:
Q1: Madrid. Q2: Spain. Q3: Championship.

Incorrect Format Example (DO NOT DO THIS):
Q1: The city of Madrid is beautiful.

Context: '{passage}'
Questions:
{questions_text}"""
        return Prompts(apathetic=apathetic, fear=fear, anger=angry, joy=joy)

    def get_record_identifier(self, record: Dict[str, Any]) -> int:
        return record["idx"]

    def postprocess_raw_response(self, raw_response: str) -> str:
        return raw_response
