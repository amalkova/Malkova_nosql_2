import os
import re

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

load_dotenv(".env")

INDEX_NAME = "arxiv-papers"
MODEL_NAME = "allenai/specter2_base"

TOP_K = 10
DISPLAY_K = 5
RRF_K = 60

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = pc.Index(INDEX_NAME)

print(f"Завантажуємо модель: {MODEL_NAME}")
model = SentenceTransformer(MODEL_NAME)

df = pd.read_parquet("data/arxiv_subset.parquet").reset_index(drop=True)


# -------------------------------------------------------------------
# Tokenization for BM25
# -------------------------------------------------------------------

def tokenize(text):
    return re.findall(r"\b\w+\b", str(text).lower())


corpus_texts = (
    df["title"].astype(str)
    + " "
    + df["abstract"].astype(str)
).tolist()

tokenized_corpus = [tokenize(text) for text in corpus_texts]

print("Будуємо BM25 index...")
bm25 = BM25Okapi(tokenized_corpus)


# -------------------------------------------------------------------
# BM25 search
# -------------------------------------------------------------------

def bm25_search(query, top_k=TOP_K):
    tokens = tokenize(query)
    scores = bm25.get_scores(tokens)

    top_indices = np.argsort(-scores)[:top_k]

    results = []

    for idx in top_indices:
        results.append(
            {
                "doc_id": int(idx),
                "score": float(scores[idx]),
            }
        )

    return results


# -------------------------------------------------------------------
# Vector search
# -------------------------------------------------------------------

def vector_search(query, top_k=TOP_K):
    vector = model.encode(
        query,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    response = index.query(
        vector=vector.tolist(),
        top_k=top_k,
        include_metadata=True,
    )

    results = []

    for match in response.get("matches", []):
        paper_id = match["id"]

        # IDs were created as paper_<row_number>
        doc_id = int(paper_id.split("_")[1])

        results.append(
            {
                "doc_id": doc_id,
                "score": float(match["score"]),
            }
        )

    return results


# -------------------------------------------------------------------
# Reciprocal Rank Fusion
# -------------------------------------------------------------------

def rrf_fusion(bm25_results, vector_results, k=60):
    scores = {}

    for rank, result in enumerate(bm25_results, start=1):
        doc_id = result["doc_id"]
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank)

    for rank, result in enumerate(vector_results, start=1):
        doc_id = result["doc_id"]
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank)

    ranked = sorted(
        scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    return [
        {
            "doc_id": doc_id,
            "score": score,
        }
        for doc_id, score in ranked
    ]


# -------------------------------------------------------------------
# Output
# -------------------------------------------------------------------

def print_results(title, results, score_name="Score", top_k=DISPLAY_K):
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)

    for rank, result in enumerate(results[:top_k], start=1):
        doc_id = result["doc_id"]
        row = df.iloc[doc_id]

        print(f"\n#{rank}")
        print(f"{score_name}: {result['score']:.6f}")
        print(f"Title: {row['title']}")
        print(f"Authors: {row['authors']}")
        print(f"Category: {row['category']}")
        print(f"Year: {row['year']}")


# -------------------------------------------------------------------
# Demonstration queries
# -------------------------------------------------------------------

queries = [
    "BERT fine-tuning",
    "Yann LeCun convolutional networks",
    "making computers understand human emotions from text",
]


for query in queries:
    print("\n\n" + "#" * 90)
    print(f"QUERY: {query}")
    print("#" * 90)

    bm25_results = bm25_search(query)
    vector_results = vector_search(query)

    hybrid_results = rrf_fusion(
        bm25_results,
        vector_results,
        k=RRF_K,
    )

    hybrid_k1 = rrf_fusion(
        bm25_results,
        vector_results,
        k=1,
    )

    print_results(
        "BM25 TOP-5",
        bm25_results,
        "BM25 score",
    )

    print_results(
        "VECTOR TOP-5",
        vector_results,
        "Vector score",
    )

    print_results(
        f"HYBRID RRF TOP-5 (k={RRF_K})",
        hybrid_results,
        "RRF score",
    )

    print_results(
        "HYBRID RRF TOP-5 (k=1)",
        hybrid_k1,
        "RRF score",
    )

    bm25_ids = [r["doc_id"] for r in bm25_results[:DISPLAY_K]]
    vector_ids = [r["doc_id"] for r in vector_results[:DISPLAY_K]]
    hybrid_ids = [r["doc_id"] for r in hybrid_results[:DISPLAY_K]]

    print("\nCOMPARISON")
    print("BM25 top-5 IDs:", bm25_ids)
    print("Vector top-5 IDs:", vector_ids)
    print("Hybrid top-5 IDs:", hybrid_ids)

    only_hybrid = [
        doc_id
        for doc_id in hybrid_ids
        if doc_id not in bm25_ids and doc_id not in vector_ids
    ]

    print(
        "Documents only in Hybrid top-5:",
        only_hybrid,
    )


print("\nDONE")
