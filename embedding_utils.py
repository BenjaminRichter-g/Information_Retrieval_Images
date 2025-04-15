import numpy as np
import sqlite3

from google import genai
from dotenv import dotenv_values
import numpy as np

# Initialize the GenAI client
config = dotenv_values(".env")
client = genai.Client(api_key=config.get("API_KEY"))

def embed_text(text, conn):
    """Generates or retrieves an embedding for a given text."""
    cursor = conn.cursor()

    # Check if the embedding already exists
    cursor.execute("SELECT gemini_embedding FROM embeddings WHERE md5 = ?", (text,))
    result = cursor.fetchone()
    if result:
        print(f"Using cached embedding for text: {text[:50]}...")
        return np.frombuffer(result[0], dtype=np.float32)

    # Generate a new embedding using Google GenAI
    response = client.models.embed_content(
        model="gemini-embedding-exp-03-07",
        contents=text
    )
    embedding = np.array(response.embeddings[0], dtype=np.float32)
    embedding_bytes = embedding.tobytes()

    # Save the embedding to the database
    cursor.execute("""
        INSERT INTO embeddings (md5, gemini_embedding)
        VALUES (?, ?)
    """, (text, embedding_bytes))
    conn.commit()

    return embedding


def cosine_similarity(embedding1, embedding2):
    """Computes cosine similarity between two embeddings."""
    if embedding1 is None or embedding2 is None:
        return 0.0
    return float(np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2)))