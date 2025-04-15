from embeddings import Embedder
import vector_db as vd

def search_top_k_embeddings(query_prompt, k=10):
    """
    Search for the top-k most similar embeddings in Milvus based on a query prompt.
    """
    # Embed the query prompt using Gemini
    embedder = Embedder()
    query_embedding = embedder.get_embedding(query_prompt)

    # Search in Milvus
    milvus_db = vd.MilvusDb()
    results = milvus_db.search(query_embedding, k=k)

    # Sort and return results
    sorted_results = sorted(results, key=lambda x: x["score"], reverse=True)
    return sorted_results


if __name__ == "__main__":
    query_prompt = "Describe what is happening in this image in one sentence."
    top_k_results = search_top_k_embeddings(query_prompt, k=10)

    print("\nTop-10 Similar Embeddings:")
    for result in top_k_results:
        print(f"MD5: {result['md5']}, Caption: {result['caption']}, Score: {result['score']}")