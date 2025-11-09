import os
import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()
model = SentenceTransformer(os.getenv("MODEL_NAME"))
client = chromadb.PersistentClient(path=os.getenv("DB_DIR"))

def search_all(query):
    emb = model.encode(query).tolist()

    results = {}
    for col_name in ["products", "reviews", "faqs"]:
        col = client.get_or_create_collection(col_name)
        match = col.query(query_embeddings=[emb], n_results=3)
        results[col_name] = match
    
    return results

def build_context(results):
     context = ""
     for col_name, r in results.items():
         documents = r.get("documents", [[]])[0]
         for doc in documents:
             context += f"\n[{col_name.upper()}]\n{doc}\n"
     return context
    
