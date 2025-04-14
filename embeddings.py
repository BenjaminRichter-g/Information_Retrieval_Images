from google import genai
from dotenv import dotenv_values

class Embedder():

    def __init__(self):
        config = dotenv_values(".env")
        self.client = genai.Client(api_key=config.get("API_KEY"))

    def get_embedding(self, content):
        try:
            result = self.client.models.embed_content(
                model="gemini-embedding-exp-03-07",
                contents=content
            )
            embedding = result.embeddings[0]  # Assuming result.embeddings is a list of lists
            embedding = list(map(float, embedding))  # Convert to a flat list of floats
            return embedding
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return None

    def batch_embeddings(self, contents):

        results = []
        for content in contents:
            try:
                results.append(self.get_embedding(content))
            except Exception as e:
                print(f"Error {e} has occured for content {content}")
        
        return results