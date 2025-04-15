from google import genai
from dotenv import dotenv_values
import numpy as np

class Embedder:

    def __init__(self):
        config = dotenv_values(".env")
        self.client = genai.Client(api_key=config.get("API_KEY"))

    def get_embedding(self, content):
        try:
            print(f"Generating embedding for content: {content}")  # Debug print
            result = self.client.models.embed_content(
                model="gemini-embedding-exp-03-07",
                contents=content
            )
            return result.embeddings
        
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return None

    def batch_embeddings(self, contents):
        results = []
        for content in contents:
            try:
                embedding = self.get_embedding(content)
                if embedding is None or not isinstance(embedding, list):
                    print(f"Invalid embedding for content: {content}. Skipping...")
                    results.append(None)
                else:
                    results.append(embedding)
            except Exception as e:
                print(f"Error {e} has occurred for content {content}")
                results.append(None)
        return results
    
    def double_embedding_test(self, gemini_caption, hf_caption):
        try:
            gemini_embedding = self.get_embedding(gemini_caption)
            hf_embedding = self.get_embedding(hf_caption)
            if isinstance(gemini_embedding, list):
                gemini_embed = np.array(gemini_embed)
            if isinstance(hf_embedding, list):
                hf_embed = np.array(hf_embed)
            return gemini_embedding, hf_embedding
        except Exception as e:
            print(f"Error generating double embedding: {e}")