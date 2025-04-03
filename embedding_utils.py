import numpy as np
from gemini_api import ModelApi
"""
Embed any text using Gemini's embedding model

Compute cosine similarity between two captions
"""
model = ModelApi()

def embed_text(text):
    return model.embed_text(text)

def cosine_similarity(text1, text2):
    emb1 = embed_text(text1)
    emb2 = embed_text(text2)
    if emb1 is None or emb2 is None:
        return 0.0
    return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))
