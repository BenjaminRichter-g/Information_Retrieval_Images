import numpy as np
from gemini_api import ModelApi
from google import generativeai as genai
from dotenv import load_dotenv
import os
import time
"""
Embed any text using Gemini's embedding model

Compute cosine similarity between two captions
"""
genai.configure(api_key=os.getenv("API_KEY"))
model = ModelApi()

def embed_text(text):
    try:
        # Ensure the input is a single string
        if not isinstance(text, str):
            raise ValueError(f"Expected a single string, but got {type(text)}: {text}")

        response = genai.embed_content(
            model="models/embedding-001",
            content=text,
            task_type="retrieval_document"
        )
        embedding = np.array(response["embedding"])
        print(f"Generated embedding for text: {text[:50]}... -> {embedding[:5]}")  # Debug print

        # Add a delay to avoid exceeding the quota
        time.sleep(5)  # Adjust the delay as needed (e.g., 0.5 seconds for ~120 requests per minute)
        return embedding
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return np.zeros(3072)  # Return a zero vector as a placeholder
    
def cosine_similarity(text1, text2):
    emb1 = embed_text(text1)
    emb2 = embed_text(text2)
    print(f"emb1: {emb1[:5] if emb1 is not None else None}, emb2: {emb2[:5] if emb2 is not None else None}")  # Debug print
    if emb1 is None or emb2 is None:
        return 0.0
    return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))