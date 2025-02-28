from typing import Dict, List
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import structlog
from pydantic import BaseModel
from transformers import QuantoConfig
from retrieval.vector_db import search
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


def format_utterance(message: str, context: str, role: str) -> str:
    return f"<turn><role>{role}</role>\n{context}\n{message}\n</turn>\n"


def format_multiturn_conversation(iteration: ChatIteration, context: str) -> str:
    user_input: str = format_utterance(
        message=iteration.message, context=context, role="user"
    )

    history: str = "".join(
        [
            format_utterance(message=history_entry.message, role=history_entry.role)
            for history_entry in iteration.history
        ]
    )

    prompt: str = f"{history}{user_input}\n<turn><role>assistant</role>\n"
    return prompt


def generate(iteration: ChatIteration) -> str:

    logger.info("Doing RAG...")
    results: List[Dict[str, str]] = search(question=iteration.message)
    if len(results) > 0:
        logger.info("RAG founds results!")
        context: str = "\n\n".join([r["chunk_raw"] for r in results])
    else:
        logger.info("RAG found no results")
        context: str = ""

    logger.info("Embedding request")
    prompt: str = format_multiturn_conversation(iteration=iteration, context=context)
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
