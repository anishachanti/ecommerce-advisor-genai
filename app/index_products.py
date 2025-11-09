import os
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

model = SentenceTransformer(os.getenv("MODEL_NAME"))
client = chromadb.PersistentClient(path=os.getenv("DB_DIR"))
products_col = client.get_or_create_collection("products")

df = pd.read_csv("../data/products.csv")

for _, row in df.iterrows():
    text = f"""
    {row['name']} by {row['brand']}
    Category: {row['category']}
    Price: {row['price']}
    Description: {row['long_description']}
    """
    emb = model.encode(text).tolist()
    products_col.add(
        ids=[row['product_id']],
        documents=[text],
        metadatas=[{"product_id": row['product_id'], "name": row['name']}],
        embeddings=[emb]
    )

print("✅ Products indexed successfully")
