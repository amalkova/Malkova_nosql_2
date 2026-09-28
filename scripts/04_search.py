import os
from datetime import datetime

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

load_dotenv(".env")

INDEX_NAME = "arxiv-papers"
MODEL_NAME = "allenai/specter2_base"
TOP_K = 5

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = pc.Index(INDEX_NAME)

print(f"Завантажуємо модель: {MODEL_NAME}")
model = SentenceTransformer(MODEL_NAME)

df = pd.read_parquet("data/arxiv_subset.parquet")
embeddings = np.load("embeddings/embeddings.npy")


def encode_query(query: str) -> np.ndarray:
    return model.encode(
        query,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )


def print_pinecone_results(title, result):
    print(f"\n{'=' * 80}")
    print(title)
    print("=" * 80)

    matches = result.get("matches", [])

    if not matches:
        print("Результатів не знайдено.")
        return

    for rank, match in enumerate(matches, start=1):
        metadata = match.get("metadata", {})

        print(f"\n#{rank}")
        print(f"Score: {match.get('score'):.6f}")
        print(f"Title: {metadata.get('title')}")
        print(f"Category: {metadata.get('category')}")
        print(f"Year: {metadata.get('year')}")
        print(f"Abstract: {metadata.get('abstract', '')[:300]}...")


def pinecone_search(query, filter_dict=None):
    query_vector = encode_query(query)

    return index.query(
        vector=query_vector.tolist(),
        top_k=TOP_K,
        include_metadata=True,
        filter=filter_dict,
    )


# -------------------------------------------------------------------
# 1. Чистий семантичний пошук
# -------------------------------------------------------------------

semantic_query = "teaching machines to recognize objects in pictures"

result = pinecone_search(semantic_query)

print_pinecone_results(
    f"SEMANTIC SEARCH\nQuery: {semantic_query}",
    result,
)


# -------------------------------------------------------------------
# 2A. Пошук з фільтрами:
# reinforcement learning + cs.LG + останні 5 років
# -------------------------------------------------------------------

current_year = datetime.now().year
recent_year = current_year - 5

filtered_query_a = "reinforcement learning"

filter_a = {
    "$and": [
        {"category": {"$eq": "cs.LG"}},
        {"year": {"$gte": recent_year}},
    ]
}

result_a = pinecone_search(
    filtered_query_a,
    filter_a,
)

print_pinecone_results(
    (
        "FILTERED SEARCH A\n"
        f"Query: {filtered_query_a}\n"
        f"category = cs.LG, year >= {recent_year}"
    ),
    result_a,
)


# -------------------------------------------------------------------
# 2B. Старі статті до 2015 року
# -------------------------------------------------------------------

filtered_query_b = "machine learning methods"

filter_b = {
    "year": {"$lt": 2015}
}

result_b = pinecone_search(
    filtered_query_b,
    filter_b,
)

print_pinecone_results(
    (
        "FILTERED SEARCH B\n"
        f"Query: {filtered_query_b}\n"
        "year < 2015"
    ),
    result_b,
)


# -------------------------------------------------------------------
# 3. Локальне порівняння similarity metrics
# -------------------------------------------------------------------

metric_query = "teaching machines to recognize objects in pictures"
q = encode_query(metric_query)

# Embeddings і query нормалізовані,
# тому cosine similarity фактично дорівнює dot product.
cosine_scores = embeddings @ q
dot_scores = embeddings @ q

# Для L2 менша відстань = кращий результат.
l2_distances = np.linalg.norm(embeddings - q, axis=1)

cosine_top = np.argsort(-cosine_scores)[:TOP_K]
dot_top = np.argsort(-dot_scores)[:TOP_K]
l2_top = np.argsort(l2_distances)[:TOP_K]


def print_local_results(name, indices, scores, higher_is_better=True):
    print(f"\n{'=' * 80}")
    print(name)
    print("=" * 80)

    for rank, idx in enumerate(indices, start=1):
        row = df.iloc[idx]
        value = scores[idx]

        label = "Similarity" if higher_is_better else "Distance"

        print(f"\n#{rank}")
        print(f"{label}: {value:.6f}")
        print(f"Title: {row['title']}")
        print(f"Category: {row['category']}")
        print(f"Year: {row['year']}")


print_local_results(
    "LOCAL COSINE SIMILARITY",
    cosine_top,
    cosine_scores,
)

print_local_results(
    "LOCAL DOT PRODUCT",
    dot_top,
    dot_scores,
)

print_local_results(
    "LOCAL L2 DISTANCE",
    l2_top,
    l2_distances,
    higher_is_better=False,
)


# -------------------------------------------------------------------
# 4. Порівняння ranking
# -------------------------------------------------------------------

print(f"\n{'=' * 80}")
print("METRIC COMPARISON")
print("=" * 80)

print("Cosine top-5 indexes:", cosine_top.tolist())
print("Dot product top-5 indexes:", dot_top.tolist())
print("L2 top-5 indexes:", l2_top.tolist())

print(
    "Cosine == Dot:",
    np.array_equal(cosine_top, dot_top),
)

print(
    "Cosine == L2:",
    np.array_equal(cosine_top, l2_top),
)
