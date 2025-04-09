import numpy as np
from gemini_api import ModelApi
from google import generativeai as genai
from dotenv import load_dotenv
import os
"""
Embed any text using Gemini's embedding model

Compute cosine similarity between two captions
"""
load_dotenv()
genai.configure(api_key=os.getenv("API_KEY"))
model = ModelApi()

def embed_text(text):
    try:
        response = genai.embed_content(
            model="models/embedding-001",
            content=text,
            task_type="retrieval_document"
        )
        return np.array(response["embedding"])
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None
def cosine_similarity(text1, text2):
    emb1 = embed_text(text1)
    emb2 = embed_text(text2)
    if emb1 is None or emb2 is None:
        return 0.0
    return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))
