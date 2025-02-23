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
        output_dir="./velvet_lora",
        evaluation_strategy="steps",  # Valutazione più frequente per fermare il training prima
        eval_steps=500,  # Frequenza della valutazione
        save_strategy="steps",  # Salvataggio più frequente
        save_steps=1000,  # Salva ogni 1000 step
        save_total_limit=2,  # Mantieni solo gli ultimi 2 checkpoint per risparmiare spazio
        per_device_train_batch_size=1,  # Riduci il batch per evitare memory overload su Mac
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,  # Simula batch più grandi senza pesare sulla RAM
        num_train_epochs=5,  # Riduci il numero di epoche per velocizzare il training
        logging_dir="./logs",
        logging_steps=200,  # Frequenza dei log
        fp16=False,  # NO FP16 su MPS! Usa invece bf16 (se disponibile)
        bf16=True,  # Usa bfloat16 se disponibile
        optim="adamw_torch",  # Ottimizzatore più efficiente su MPS
        warmup_ratio=0.1,  # Warmup per stabilizzare l'inizio del training
        report_to="none",  # Disabilita reporting automatico (come WandB)
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

    model.save_pretrained("./lora_fine_tuned_model")
    TOKENIZER.save_pretrained("./lora_fine_tuned_model")

    input_text = "<s>Utente: Quali sono i principali obiettivi dell'intelligenza artificiale?</s>\n<s>Assistente: "
    embeddings = TOKENIZER(input_text, return_tensors="pt").to("mps")
    output = model.generate(embeddings["input_ids"], max_new_tokens=500)
    print(TOKENIZER.decode(output[0]))


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
