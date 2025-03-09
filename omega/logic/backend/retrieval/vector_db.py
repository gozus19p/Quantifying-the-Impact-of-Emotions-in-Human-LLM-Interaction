import os
from typing import Any, Dict, List

from pymilvus import AnnSearchRequest, MilvusClient, RRFRanker
import structlog

from retrieval.embedding import embed


# __LOCAL_DB: str = os.path.join(os.path.dirname(__file__), "milvus.db")
__CLIENT: MilvusClient = MilvusClient(uri="http://localhost:19530")
__LOGGER: structlog.BoundLogger = structlog.get_logger()


def overlap_chunk(
    text: str, max_chars: int = 500, max_words: int = 250, overlap_words: int = 20
) -> List[str]:
    words = text.split()
    chunks = []
    i = 0

    while i < len(words):
        chunk_words = words[i : i + max_words]  # Prende max_words parole
        chunk = " ".join(chunk_words)

        # Se il chunk supera max_chars, riduci gradualmente la lunghezza
        while len(chunk) > max_chars and len(chunk_words) > 1:
            chunk_words.pop()  # Rimuove parole finché il chunk non è sotto 768 caratteri
            chunk = " ".join(chunk_words)

        chunks.append(chunk)
        i += (
            max_words - overlap_words
        )  # Offset per garantire overlap con le ultime 15 parole

    return chunks


def insert_lecture(lecture_text: str, lecture_no: int) -> bool:
    __LOGGER.info(f"Storing lecture {lecture_no}")
    __LOGGER.info("Do chunking...")
    chunks: List[str] = overlap_chunk(lecture_text)
    __LOGGER.info(f"Done {len(chunks)} chunks")

    __LOGGER.info("Embedding...")
    embeddings: List[List[float]] = embed(chunks)
    __LOGGER.info("Done")

    counter: int = 0
    for chunk, embedding in zip(chunks, embeddings):
        counter += 1
        id_: str = f"{lecture_no}-{counter}"
        __LOGGER.info(f"Inserting {id_}")
        __CLIENT.insert(
            collection_name="lectures",
            data={
                "id": id_,
                "chunk": embedding,
                "chunk_raw": chunk,
                "lecture": lecture_text,
                "lecture_no": lecture_no,
            },
        )


def __parse_result(result: Dict[str, Any]) -> Dict[str, str]:
    return {
        "id": result["id"],
        "distance": result["distance"],
        "lecture_no": result["entity"]["lecture_no"],
        "chunk_raw": result["entity"]["chunk_raw"],
    }


def __z_score(score: float) -> float:
    mean_score: float = 0.7703366847832998
    std_score: float = 0.021180900917752647
    return (score - mean_score) / std_score


def __filter_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [r for r in results if __z_score(score=r["distance"]) > 3]


def search(question: str, threshold: float = 0.016, k: int = 60) -> List[Dict]:
    query: str = f"{question}"
    query_vector: List[float] = embed([query])[0]
    search_result: List[List[Dict[str, Any]]] = __CLIENT.search(
        collection_name="lectures",
        data=[query_vector],
        search_params={"metric_type": "COSINE"},
        limit=3,
        anns_field="chunk",
        output_fields=["lecture_no", "chunk_raw"],
    )
    # search_result: List[List[Dict[str, Any]]] = __CLIENT.hybrid_search(
    #     collection_name="lectures",
    #     reqs=[
    #         AnnSearchRequest(
    #             anns_field="chunk",
    #             data=[query_vector],
    #             param={"metric_type": "COSINE"},
    #             limit=3,
    #         )
    #     ],
    #     limit=3,
    #     ranker=RRFRanker(k=k),
    #     output_fields=["lecture_no", "chunk_raw"],
    # )

    results: List[Dict[str, Any]] = [__parse_result(result=r) for r in search_result[0]]
    filtered_results: List[Dict[str, Any]] = __filter_results(results=results)
    return filtered_results


# import json

# print(
#     json.dumps(search("Come posso classificare le intelligenze artificiali?"), indent=2)
# )


# with open(
#     "omega/logic/backend/retrieval/resources/post-processed/trascrizione lezione 1.txt",
#     "r",
# ) as f:
#     insert_lecture(lecture_text=f.read(), lecture_no=1)
#     pass
