import os
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv(".env")

INPUT_PARQUET = "data/arxiv_subset.parquet"
INPUT_EMBEDDINGS = "embeddings/embeddings.npy"

INDEX_NAME = "arxiv-papers"
VECTOR_DIM = 768
BATCH_SIZE = 200

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

# Створюємо індекс, якщо його ще немає
if INDEX_NAME not in pc.list_indexes().names():
    print(f"Створюємо індекс {INDEX_NAME}...")

    pc.create_index(
        name=INDEX_NAME,
        dimension=VECTOR_DIM,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1",
        ),
    )

    while not pc.describe_index(INDEX_NAME).status["ready"]:
        print("Очікуємо готовність індексу...")
        time.sleep(2)
else:
    print(f"Індекс {INDEX_NAME} вже існує.")

index = pc.Index(INDEX_NAME)

print("Читаємо дані...")
df = pd.read_parquet(INPUT_PARQUET)
embeddings = np.load(INPUT_EMBEDDINGS)

if len(df) != len(embeddings):
    raise ValueError(
        f"Кількість статей ({len(df)}) не збігається "
        f"з кількістю ембеддингів ({len(embeddings)})"
    )

print(f"Статей: {len(df)}")
print(f"Embeddings shape: {embeddings.shape}")

for start in tqdm(
    range(0, len(df), BATCH_SIZE),
    desc="Завантаження в Pinecone",
):
    end = min(start + BATCH_SIZE, len(df))

    batch = []

    for i in range(start, end):
        row = df.iloc[i]

        batch.append(
            {
                "id": f"paper_{i}",
                "values": embeddings[i].tolist(),
                "metadata": {
                    "arxiv_id": str(row["id"]),
                    "title": str(row["title"]),
                    "abstract": str(row["abstract"])[:500],
                    "authors": str(row["authors"])[:200],
                    "year": int(row["year"]),
                    "category": str(row["category"]),
                },
            }
        )

    index.upsert(vectors=batch)

print("\nЗавантаження завершено.")

time.sleep(3)

stats = index.describe_index_stats()

print("Статистика індексу:")
print(stats)
print(f"Загальна кількість векторів: {stats.total_vector_count}")
