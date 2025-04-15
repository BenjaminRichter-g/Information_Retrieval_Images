import json
import csv
import numpy as np
from embedding_utils import embed_text, cosine_similarity
from embedding_utils import embed_text, cosine_similarity
import vector_db as vd


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

def evaluate_post_testing(gemini_path, other_model_path, reference_path, output_csv):
    # Load captions
    try:
        with open(gemini_path, "r") as f:
            gemini_captions = json.load(f)
        print(f"Loaded {len(gemini_captions)} Gemini captions.")
    except Exception as e:
        print(f"Error loading Gemini captions from {gemini_path}: {e}")
        return

    try:
        with open(other_model_path, "r") as f:
            other_model_captions = json.load(f)
        print(f"Loaded {len(other_model_captions)} Hugging Face captions.")
    except Exception as e:
        print(f"Error loading Hugging Face captions from {other_model_path}: {e}")
        return

    try:
        with open(reference_path, "r") as f:
            reference_captions = json.load(f)
        print(f"Loaded {len(reference_captions)} reference captions.")
    except Exception as e:
        print(f"Error loading reference captions from {reference_path}: {e}")
        return

    results = []

    # Process each Gemini caption
    for idx, (filename, gemini_caption) in enumerate(gemini_captions.items(), start=1):
        print(f"Processing {idx}/{len(gemini_captions)}: {filename}")
        references = reference_captions.get(filename, [])
        other_caption = other_model_captions.get(filename, "")

        if not references or not other_caption:
            print(f"Skipping {filename}: Missing references or Hugging Face caption.")
            continue

        print(f"Gemini caption: {gemini_caption}")
        print(f"References: {references}")
        print(f"Hugging Face caption: {other_caption}")

        # Combine multiple captions into a single string
        gemini_caption = " ".join(gemini_caption) if isinstance(gemini_caption, list) else gemini_caption
        references = [" ".join(ref) if isinstance(ref, list) else ref for ref in references]

        # Embed Gemini caption
        print(f"Generating embedding for Gemini caption: {gemini_caption[:50]}...")
        gemini_embedding = embed_text(gemini_caption)
        if gemini_embedding is None or np.all(gemini_embedding == 0):
            print(f"Skipping invalid Gemini embedding for {filename}.")
            continue

        # Embed references
        reference_embeddings = []
        for i, ref in enumerate(references, start=1):
            print(f"Generating embedding {i}/{len(references)} for reference: {ref[:50]}...")
            ref_embedding = embed_text(ref)
            if ref_embedding is None or np.all(ref_embedding == 0):
                print(f"Skipping invalid embedding for reference: {ref}")
                continue
            reference_embeddings.append(ref_embedding)

        if not reference_embeddings:
            print(f"Skipping {filename}: All reference embeddings are invalid.")
            continue

        # Compute cosine similarity scores for Gemini captions
        gemini_scores = []
        for ref_emb in reference_embeddings:
            if gemini_embedding.shape != ref_emb.shape:
                print(f"Skipping comparison due to shape mismatch: {gemini_embedding.shape} vs {ref_emb.shape}")
                continue
            gemini_scores.append(cosine_similarity(gemini_embedding, ref_emb))

        gemini_similarity_max = max(gemini_scores) if gemini_scores else 0
        gemini_similarity_avg = sum(gemini_scores) / len(gemini_scores) if gemini_scores else 0

        # Embed the other model's caption
        print(f"Generating embedding for Hugging Face caption: {other_caption[:50]}...")
        other_embedding = embed_text(other_caption)
        if other_embedding is None or np.all(other_embedding == 0):
            print(f"Skipping invalid embedding for Hugging Face caption in {filename}.")
            continue

        # Compute cosine similarity scores for the other model's caption
        other_model_scores = []
        for ref_emb in reference_embeddings:
            if other_embedding.shape != ref_emb.shape:
                print(f"Skipping comparison due to shape mismatch: {other_embedding.shape} vs {ref_emb.shape}")
                continue
            other_model_scores.append(cosine_similarity(other_embedding, ref_emb))

        other_model_similarity_max = max(other_model_scores) if other_model_scores else 0
        other_model_similarity_avg = sum(other_model_scores) / len(other_model_scores) if other_model_scores else 0

        # Append results
        results.append({
            "image": filename,
            "gemini_caption": gemini_caption,
            "other_model_caption": other_caption,
            "gemini_similarity_max": round(gemini_similarity_max, 4),
            "gemini_similarity_avg": round(gemini_similarity_avg, 4),
            "other_model_similarity_max": round(other_model_similarity_max, 4),
            "other_model_similarity_avg": round(other_model_similarity_avg, 4),
        })

    # Save results to CSV
    if not results:
        print("No results to write to the CSV file.")
        return

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"Saved post-testing evaluation results to {output_csv}")