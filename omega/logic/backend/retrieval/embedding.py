from typing import Any, List
import torch.nn.functional as F

from torch import Tensor
from transformers import AutoTokenizer, AutoModel


TOKENIZER: AutoTokenizer = AutoTokenizer.from_pretrained(
    "intfloat/multilingual-e5-base"
)
MODEL: AutoModel = AutoModel.from_pretrained("intfloat/multilingual-e5-base")


def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def embed(input_texts: List[str]) -> List[List[float]]:

    batch_dict = TOKENIZER(
        input_texts, max_length=512, padding=True, truncation=True, return_tensors="pt"
    )

    outputs = MODEL(**batch_dict)
    embeddings = average_pool(outputs.last_hidden_state, batch_dict["attention_mask"])
    return embeddings.tolist()
