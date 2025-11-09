from fastapi import FastAPI
from rag_engine import search_all, build_context
from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
app = FastAPI()

@app.get("/chat")
def chat(q: str):
    results = search_all(q)
    context = build_context(results)
    prompt = f"""
You are a friendly and smart e-commerce advisor.
Use ONLY the information provided. Do not make up specifications.

User Question: {q}

Product Information:
{context}

Give a clear, concise recommendation.
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "You are a helpful shopping assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )

    answer = response.choices[0].message.content

    return {
        "question": q,
        "answer": answer,
    }
