from inspect import getmembers
import json
import token
from typing import Any, Dict, List
from h11 import Data
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import load_dataset, Dataset
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
import torch
import structlog
import os

os.environ["TOKENIZERS_PARALLELISM"] = "true"

logger = structlog.getLogger()


model_name: str = "Almawave/Velvet-14B"

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


def train() -> None:

    training_args: TrainingArguments = TrainingArguments(
        output_dir="./lora_fine_tuned_model",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=3,
        logging_dir="./logs",
        logging_steps=500,
        save_total_limit=2,
        bf16=True,  # Usa la mezza precisione per migliorare l'efficienza
    )

    tokenized_dataset: Dataset = get_dataset()["train"].train_test_split(
        test_size=0.2, seed=42
    )

    model: PeftModel = get_peft_model_for_training()
    trainer: Trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=TOKENIZER,
    )
    trainer.train()

    input_text = "Quali sono i principali obiettivi dell'intelligenza artificiale?"
    input_ids = TOKENIZER(input_text, return_tensors="pt").input_ids
    output = model.generate(input_ids, max_length=100)
    print(TOKENIZER.decode(output[0], skip_special_tokens=True))

    model.save_pretrained("./lora_fine_tuned_model")
    TOKENIZER.save_pretrained("./lora_fine_tuned_model")


def get_dataset() -> Dataset:
    dataset: Dataset = load_dataset(
        "json",
        data_files=os.path.join(
            os.path.dirname(__file__), "resources/training data/lezione1.jsonl"
        ),
    )
    formatted_dataset: Dataset = dataset.map(format_velvet_conversation)

    tokenized_dataset: Dataset = formatted_dataset.map(tokenize, batched=True)
    return tokenized_dataset


def get_peft_model_for_training() -> PeftModel:
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,  # Grado di riduzione (minore = più efficiente)
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],  # Moduli su cui applicare LoRA
    )
    model: PeftModel = get_peft_model(VELVET, lora_config)
    return model


def format_velvet_conversation(example: Dict[str, str]):
    return {
        "input_text": f"<s>Utente: {example['question']}</s>\n<s>Assistente: ",
        "target_text": f"{example['answer']}</s>",
    }


def tokenize(example: Dict[str, str]) -> Dict[str, Any]:
    inputs = TOKENIZER(
        example["input_text"], padding="max_length", truncation=True, max_length=512
    )
    targets = TOKENIZER(
        example["target_text"], padding="max_length", truncation=True, max_length=512
    )
    inputs["labels"] = targets["input_ids"]
    return inputs


if __name__ == "__main__":
    train()
