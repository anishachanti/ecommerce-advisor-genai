import os
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

model = SentenceTransformer(os.getenv("MODEL_NAME"))
client = chromadb.PersistentClient(path=os.getenv("DB_DIR"))
reviews_col = client.get_or_create_collection("reviews")

df = pd.read_csv("../data/products.csv")

for _, row in df.iterrows():
    review = row["reviews_text"]
    emb = model.encode(review).tolist()
    reviews_col.add(
        ids=[row["product_id"]],
        documents=[review],
        metadatas=[{"product_id": row["product_id"]}]
    , embeddings=[emb])

print("✅ Reviews indexed successfully")
