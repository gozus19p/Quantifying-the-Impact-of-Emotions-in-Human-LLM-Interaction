from typing import Dict, List
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
import torch
import structlog
from pydantic import BaseModel
from transformers import QuantoConfig
from retrieval.vector_db import search
import os
from threading import Thread

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


def format_utterance(message: str, role: str) -> str:
    return f"<turn><role>{role}</role>\n{message}\n</turn>\n"


def format_multiturn_conversation(iteration: ChatIteration, context: str) -> str:
    if context:
        prompt: str = (
            f"Dato un contesto, rispondi alla domanda. Contesto:\n\n{context}\nDomanda: {iteration.message}"
        )
    else:
        context: str = ""
        prompt: str = iteration.message

    user_input: str = format_utterance(message=prompt, role="user")

    history: str = "".join(
        [
            format_utterance(message=history_entry.message, role=history_entry.role)
            for history_entry in iteration.history
        ]
    )

    prompt: str = f"{history}{user_input}\n<turn><role>assistant</role>\n"
    return prompt


def generate_streaming(iteration: ChatIteration):
    logger.info("Doing RAG...")
    results: List[Dict[str, str]] = search(question=iteration.message)
    if len(results) > 0:
        logger.info("RAG found results!")
        context = "\n\n".join([r["chunk_raw"] for r in results]) if results else ""
    else:
        logger.info("RAG found nothing")
        context = None

    logger.info("Embedding request")
    prompt = format_multiturn_conversation(iteration=iteration, context=context)
    embeddings = TOKENIZER(prompt, return_tensors="pt").to("mps")

    logger.info("Setting up streamer")
    streamer = TextIteratorStreamer(
        TOKENIZER, skip_special_tokens=True, skip_prompt=True
    )

    # Esegui la generazione in un thread separato per non bloccare il flusso principale
    def generate_text():
        with torch.no_grad():
            VELVET.generate(
                input_ids=embeddings["input_ids"],
                max_new_tokens=500,
                attention_mask=embeddings["attention_mask"],
                streamer=streamer,
            )

    thread = Thread(target=generate_text)
    thread.start()

    # Restituisci lo streaming man mano che i token vengono generati
    # Ignora i primi `prompt_tokens` token
    for token in streamer:
        yield token
