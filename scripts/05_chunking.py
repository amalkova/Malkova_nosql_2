import os
import re
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

load_dotenv(".env")

MODEL_NAME = "allenai/specter2_base"
VECTOR_DIM = 768

FIXED_INDEX = "arxiv-chunks-fixed"
SEMANTIC_INDEX = "arxiv-chunks-semantic"

FIXED_SIZE = 100
OVERLAP = 20
SEMANTIC_MAX_WORDS = 100

BATCH_SIZE = 100

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
model = SentenceTransformer(MODEL_NAME)

df = pd.read_parquet("data/arxiv_subset.parquet")

# Беремо 30 найдовших анотацій
df["abstract_word_count"] = df["abstract"].str.split().str.len()
longest = df.nlargest(30, "abstract_word_count").copy()


def fixed_size_chunks(text, chunk_size=100, overlap=20):
    words = text.split()
    chunks = []

    start = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])

        if chunk:
            chunks.append(chunk)

        if end == len(words):
            break

        start += chunk_size - overlap

    return chunks


def semantic_chunks(text, max_words=100):
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())

    chunks = []
    current = []
    current_words = 0

    for sentence in sentences:
        sentence = sentence.strip()

        if not sentence:
            continue

        sentence_words = len(sentence.split())

        if current and current_words + sentence_words > max_words:
            chunks.append(" ".join(current))
            current = []
            current_words = 0

        current.append(sentence)
        current_words += sentence_words

    if current:
        chunks.append(" ".join(current))

    return chunks


def prepare_chunks(strategy):
    result = []

    for _, row in longest.iterrows():
        if strategy == "fixed":
            chunks = fixed_size_chunks(row["abstract"])
        else:
            chunks = semantic_chunks(row["abstract"])

        for chunk_number, chunk in enumerate(chunks):
            result.append(
                {
                    "arxiv_id": str(row["id"]),
                    "title": str(row["title"]),
                    "chunk": chunk,
                    "chunk_number": chunk_number,
                    "year": int(row["year"]),
                    "category": str(row["category"]),
                }
            )

    return result


def create_index(name):
    if name not in pc.list_indexes().names():
        print(f"Створюємо індекс {name}...")

        pc.create_index(
            name=name,
            dimension=VECTOR_DIM,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1",
            ),
        )

        while not pc.describe_index(name).status["ready"]:
            time.sleep(2)

    return pc.Index(name)


def upload_chunks(index, chunks, prefix):
    texts = [item["chunk"] for item in chunks]

    print(f"Створюємо {len(texts)} embeddings...")

    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    for start in tqdm(
        range(0, len(chunks), BATCH_SIZE),
        desc=f"Upload {prefix}",
    ):
        end = min(start + BATCH_SIZE, len(chunks))

        batch = []

        for i in range(start, end):
            item = chunks[i]

            batch.append(
                {
                    "id": f"{prefix}_{i}",
                    "values": embeddings[i].tolist(),
                    "metadata": {
                        "arxiv_id": item["arxiv_id"],
                        "title": item["title"],
                        "chunk": item["chunk"][:1000],
                        "chunk_number": item["chunk_number"],
                        "year": item["year"],
                        "category": item["category"],
                    },
                }
            )

        index.upsert(vectors=batch)


def search_chunks(index, query, name):
    query_vector = model.encode(
        query,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    result = index.query(
        vector=query_vector.tolist(),
        top_k=5,
        include_metadata=True,
    )

    print("\n" + "=" * 80)
    print(name)
    print(f"Query: {query}")
    print("=" * 80)

    for rank, match in enumerate(result.get("matches", []), start=1):
        metadata = match.get("metadata", {})

        print(f"\n#{rank} score={match.get('score'):.4f}")
        print(f"Title: {metadata.get('title')}")
        print(f"Chunk: {metadata.get('chunk', '')[:300]}...")


fixed_chunks = prepare_chunks("fixed")
semantic_chunks_data = prepare_chunks("semantic")

print(f"Fixed-size chunks: {len(fixed_chunks)}")
print(f"Semantic chunks: {len(semantic_chunks_data)}")

fixed_index = create_index(FIXED_INDEX)
semantic_index = create_index(SEMANTIC_INDEX)

upload_chunks(fixed_index, fixed_chunks, "fixed")
upload_chunks(semantic_index, semantic_chunks_data, "semantic")

time.sleep(3)

queries = [
    "machine learning methods",
    "quantum systems and particles",
]

for query in queries:
    search_chunks(
        fixed_index,
        query,
        "FIXED-SIZE CHUNKING",
    )

    search_chunks(
        semantic_index,
        query,
        "SEMANTIC CHUNKING",
    )

print("\nDONE")
