from inspect import getmembers
import json
import token
from typing import Any, Dict, List
from cycler import V
from h11 import Data
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import load_dataset, Dataset
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
import torch
import structlog
import os

os.environ["TOKENIZERS_PARALLELISM"] = "true"

logger = structlog.getLogger()


def test(model_name: str) -> str:
    logger.info(f"Loading {model_name} tokenizer")
    TOKENIZER: AutoTokenizer = AutoTokenizer.from_pretrained(model_name)
    logger.info(f"{model_name} tokenizer loaded")

    logger.info(f"Loading {model_name} model")
    VELVET: AutoModelForCausalLM = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="mps",
        torch_dtype="auto",
        # quantization_config=quantization_config,
    )
    logger.info(f"{model_name} loaded")
    input_text = "<turn><role>user</role>\nCiao. Come ti senti oggi?\n</turn>\n<turn><role>assistant</role>"
    embeddings = TOKENIZER(input_text, return_tensors="pt", padding="longest").to("mps")
    output = VELVET.generate(
        embeddings["input_ids"],
        max_new_tokens=100,
        attention_mask=embeddings["attention_mask"],
    )
    return TOKENIZER.decode(output[0], skip_special_tokens=True)


# model_name: str = os.path.dirname(__file__) + "/../lora_fine_tuned_model"
model_name: str = "Almawave/Velvet-14B"


if __name__ == "__main__":

    print(test("Almawave/Velvet-14B"))
    print(test(os.path.dirname(__file__) + "/../../Velvet-14B-joy/checkpoint-35"))
