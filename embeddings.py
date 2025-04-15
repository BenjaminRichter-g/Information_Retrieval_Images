from google import genai
from dotenv import dotenv_values
import time

class Embedder():

    def __init__(self):
        config = dotenv_values(".env")
        self.client = genai.Client(api_key=config.get("API_KEY"))

    def get_embedding(self, content):

        result = self.client.models.embed_content(
            model="gemini-embedding-exp-03-07",
            contents=content)

        return result.embeddings


    def batch_embeddings(self, contents):
        results = []
        nb_embed = len(contents)
        nb_done = 0
        for content in contents:
            print(f"Finished the {nb_done} out of {nb_embed}")
            print(f"Embeded caption: {content}")
            nb_done += 1
            time.sleep(4)
            try:
                results.append(self.get_embedding(content))
            except Exception as e:
                print(f"Error {e} has occured for content {content}")
                print("Existing results will be returned, please re-excute to finish remainder")
                if len(results)!=0:
                    return results
        
        return results
