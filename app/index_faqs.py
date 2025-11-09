import os
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

model = SentenceTransformer(os.getenv("MODEL_NAME"))
client = chromadb.PersistentClient(path=os.getenv("DB_DIR"))
faqs_col = client.get_or_create_collection("faqs")

df = pd.read_csv("../data/products.csv")

for _, row in df.iterrows():
    faq = row["faqs_text"]
    emb = model.encode(faq).tolist()
    faqs_col.add(
        ids=[row["product_id"]],
        documents=[faq],
        metadatas=[{"product_id": row["product_id"]}],
        embeddings=[emb]
    )

print("✅ FAQs indexed successfully")
