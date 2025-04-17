from google import genai
from dotenv import dotenv_values
import numpy as np
import time

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
            print(f"Raw API response type: {type(result.embeddings)}")  # Debug print
            #print(f"Raw API response content: {result.embeddings[0]}")  # Debug print

            # Extract the actual embedding data
            if hasattr(result.embeddings[0], 'values'):  # Check if 'values' attribute exists
                embedding = np.array(result.embeddings[0].values, dtype=np.float32)
            else:
                raise ValueError(f"Unexpected embedding format: {type(result.embeddings[0])}")

            print(f"Generated embedding shape: {embedding.shape}")  # Debug print
            return embedding
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
            time.sleep(4)  # Delay before the first API call
            gemini_embedding = self.get_embedding(gemini_caption)

            time.sleep(4)  # Delay before the second API call
            hf_embedding = self.get_embedding(hf_caption)

            if gemini_embedding is None or hf_embedding is None:
                raise ValueError("One or both embeddings are invalid.")

            # Validate embedding dimensions
            if gemini_embedding.shape[0] != 3072 or hf_embedding.shape[0] != 3072:  # Adjust dimension as needed
                raise ValueError(f"Unexpected embedding shape: Gemini - {gemini_embedding.shape}, HF - {hf_embedding.shape}")

            return gemini_embedding, hf_embedding
        except Exception as e:
            print(f"Error generating double embedding: {e}")
            return None, None