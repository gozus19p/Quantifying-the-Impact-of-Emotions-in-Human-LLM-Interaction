from typing import List
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import structlog
from pydantic import BaseModel
from transformers import QuantoConfig
import os

os.environ["TOKENIZERS_PARALLELISM"] = "true"

quantization_config = QuantoConfig(
    weights="int8",
)

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


class ChatHistoryEntry(BaseModel):
    role: str
    message: str


class ChatIteration(BaseModel):
    message: str
    history: List[ChatHistoryEntry]


class ChatResponse(BaseModel):
    message: str


def format_multiturn_conversation(iteration: ChatIteration) -> str:
    user_input: str = f"<s>Utente: {iteration.message}</s>\n"

    history: str = ""

    for history_entry in iteration.history:
        history += f"<s>{history_entry.role}: {history_entry.message}</s>\n"

    # Aggiunge l'ultimo turno vuoto per la generazione
    prompt: str = f"{history}{user_input}\n<s>Assistente: "

    # Formato atteso: `Utente: Ciao, chi sei?</s><s>Assistente: `
    # Nel caso di più iterazioni si ripete: `Utente: Ciao, chi sei?</s><s>Assistente: Mario</s><s>Utente: ciao, Mario!</s><s>Assistente: `
    return prompt.removeprefix("<s>")


def generate(iteration: ChatIteration) -> str:

    logger.info("Embedding request")
    prompt: str = format_multiturn_conversation(iteration)
    logger.info("Prompt", prompt=prompt)

    embeddings: torch.Tensor = TOKENIZER(prompt, return_tensors="pt").to("mps")
    with torch.no_grad():
        logger.info("Generating answer")
        output: torch.Tensor = VELVET.generate(
            input_ids=embeddings["input_ids"],
            max_new_tokens=500,
            attention_mask=embeddings["attention_mask"],
        )

    logger.info("Decoding")
    answer: str = TOKENIZER.decode(output[0])
    logger.info("Answer", answer=answer)

    logger.info("Generation completed")
    postprocessed_answer: str = (
        answer.replace(prompt, "").replace("<s>", "").replace("</s>", "").strip()
    )
    logger.info("PP answer", postprocessed_answer=postprocessed_answer)
    return postprocessed_answer
