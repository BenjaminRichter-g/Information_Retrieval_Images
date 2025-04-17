import json
import csv
import numpy as np
from embedding_utils import embed_text, cosine_similarity
import vector_db as vd
from db import init_db


def evaluate_embedding_cosine_similarity(embeddings, output_csv="results/similarity_scores_embedding.csv"):

    all_values = []

    for index in range(len(embeddings)):
        if index >= len(embeddings) or embeddings[index] is None:
            print(f"Skipping image {index} due to missing or invalid embedding.")
            continue

        all_values.append(cosine_similarity(embeddings[index][0], embeddings[index][1]))

    if not all_values:
        print("No valid embeddings to evaluate. Skipping CSV generation.")
        return

    try:
        with open(output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["index", "cosine_similarity"])
            writer.writeheader()
            writer.writerows(all_values)
        print(f"Saved cosine similarity results to {output_csv}")
    except Exception as e:
        print(f"Error saving results to CSV: {e}")

def evaluate_top_n_similarity(embeddings, output_csv="results/similarity_scores_top_n.csv", top_n=10):

    common_scores = []
    vector_db = vd.MilvusDb()

    for index in range(len(embeddings)):
        if index >= len(embeddings) or embeddings[index] is None:
            print(f"Skipping image {index} due to missing or invalid embedding.")
            continue

        gemini_n_results = vector_db.search_by_embedding(embeddings[index][0], limit=top_n)
        hf_n_results = vector_db.search_by_embedding(embeddings[index][1], limit=top_n)
        gemini_md5s = [result[0] for result in gemini_n_results]
        hf_md5 = [result[0] for result in hf_n_results]

        common_returns = 0
        for g_md5 in gemini_md5s:
            for index in range(len(hf_md5)):
                if g_md5 == hf_md5[index]:
                    common_returns += 1
                    hf_md5.pop(index) #as all md5s are unique if one is found in common we can pop and prevent n*n
                    break

        common_scores.append(common_returns/top_n)

    if not common_scores:
        print("No valid embeddings to evaluate. Skipping CSV generation.")
        return
    try:
        with open(output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["index", "top_n_indices", "similarities"])
            writer.writeheader()
            writer.writerows(common_scores)
        print(f"Saved top N similarity results to {output_csv}")
    except Exception as e:
        print(f"Error saving results to CSV: {e}")
