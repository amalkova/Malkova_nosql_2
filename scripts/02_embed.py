import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

INPUT_FILE = "data/arxiv_subset.parquet"
OUTPUT_FILE = "embeddings/embeddings.npy"
MODEL_NAME = "allenai/specter2_base"
BATCH_SIZE = 64

os.makedirs("embeddings", exist_ok=True)

print("Читаємо датасет...")
df = pd.read_parquet(INPUT_FILE)

texts = (
    df["title"].astype(str)
    + " [SEP] "
    + df["abstract"].astype(str)
).tolist()

print(f"Кількість текстів: {len(texts)}")
print(f"Завантажуємо модель: {MODEL_NAME}")

model = SentenceTransformer(MODEL_NAME)

print("Створюємо ембеддинги...")

embeddings = model.encode(
    texts,
    batch_size=BATCH_SIZE,
    show_progress_bar=True,
    normalize_embeddings=True,
    convert_to_numpy=True,
)

print(f"\nКількість ембеддингів: {len(embeddings)}")
print(f"Розмірність: {embeddings.shape[1]}")
print(f"Норма першого ембеддингу: {np.linalg.norm(embeddings[0]):.6f}")

np.save(OUTPUT_FILE, embeddings)

print(f"Збережено в {OUTPUT_FILE}")
print(f"Shape: {embeddings.shape}")
