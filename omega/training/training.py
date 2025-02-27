from inspect import getmembers
import json
import token
from typing import Any, Dict, List
from h11 import Data
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    TrainerCallback,
)
from datasets import load_dataset, Dataset
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
import torch
from torch.utils.data import DataLoader
import structlog
import os
import glob

os.environ["TOKENIZERS_PARALLELISM"] = "true"

logger = structlog.getLogger()


class FileLoggerCallback(TrainerCallback):
    def __init__(self, emotion: str):
        self.log_file = f"training-{emotion}.log"

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            with open(self.log_file, "a") as f:
                f.write(str(logs) + "\n")


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


# Trova l'ultimo checkpoint disponibile
def get_latest_checkpoint(output_dir: str):
    checkpoints = sorted(glob.glob(f"{output_dir}/checkpoint-*"), key=os.path.getctime)
    return checkpoints[-1] if checkpoints else None


def train(emotion: str) -> None:

    output_dir: str = f"./Velvet-14B-{emotion}"

    training_args: TrainingArguments = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="steps",  # Valutazione più frequente per fermare il training prima
        eval_steps=5,  # Valuta meno spesso per ridurre overhead
        save_steps=5,  # Salva meno frequentemente per non intasare il disco
        save_strategy="steps",  # Salvataggio più frequente
        save_total_limit=2,  # Mantieni solo gli ultimi 2 checkpoint per risparmiare spazio
        per_device_eval_batch_size=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,  # Accumuli meno gradienti per update
        num_train_epochs=2,  # Riduci il numero di epoche per velocizzare il training
        logging_dir="./logs",
        logging_steps=1,  # Frequenza dei log
        fp16=False,  # NO FP16 su MPS! Usa invece bf16 (se disponibile)
        bf16=True,  # Usa bfloat16 se disponibile
        optim="adamw_torch",  # Ottimizzatore più efficiente su MPS
        warmup_ratio=0.1,  # Warmup per stabilizzare l'inizio del training
        report_to="none",  # Disabilita reporting automatico (come WandB)
    )

    dataset_file_path: str = os.path.join(
        os.path.dirname(__file__), "emotions", f"{emotion}_synthetic_dataset.jsonl"
    )
    tokenized_dataset: Dataset = get_dataset(data_file_path=dataset_file_path)[
        "train"
    ].train_test_split(test_size=0.2, seed=42)

    model: PeftModel = get_peft_model_for_training()

    trainer: Trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=TOKENIZER,
        callbacks=[FileLoggerCallback(emotion=emotion)],
    )
    checkpoint_path = None
    if os.path.exists(output_dir) and os.listdir(output_dir):
        checkpoint_path: str = get_latest_checkpoint(
            output_dir=output_dir
        )  # Hugging Face Trainer cerca automaticamente il checkpoint più recente
        print(f"Riprendo il training dal checkpoint in {checkpoint_path}")

    trainer.train(resume_from_checkpoint=checkpoint_path)

    model.save_pretrained(f"./Velvet-14B-{emotion}-full")
    TOKENIZER.save_pretrained(f"./Velvet-14B-{emotion}-full-tokenizer")


def get_dataset(data_file_path: str) -> Dataset:
    dataset: Dataset = load_dataset("json", data_files=data_file_path)

    tokenized_dataset: Dataset = dataset.map(tokenize, batch_size=50, batched=True)
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


def format_conversation(messages: List[Dict[str, str]]) -> str:
    formatted_text = ""  # Inizio della sequenza

    for message in messages:
        role = message["role"]  # "user" o "assistant"
        content = message["content"]

        formatted_text += f"<turn>\n<role>{role}</role>\n{content}\n</turn>\n"

    return formatted_text


def tokenize(batch: Dict[str, List[str]]) -> Dict[str, Any]:
    formatted_texts = [format_conversation(messages) for messages in batch["messages"]]
    tokenized = TOKENIZER(
        formatted_texts, padding="max_length", truncation=True, max_length=1024
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized


if __name__ == "__main__":
    train(emotion="anger")
