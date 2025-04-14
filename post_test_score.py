import json
import csv
from embedding_utils import cosine_similarity

from embedding_utils import embed_text, cosine_similarity

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

    for filename, gemini_caption in gemini_captions.items():
        references = reference_captions.get(filename, [])
        other_caption = other_model_captions.get(filename, "")  # Use the single caption

        if not references or not other_caption:
            print(f"Skipping {filename}: Missing references or Hugging Face caption.")
            continue

        print(f"Processing {filename}:")
        print(f"Gemini caption: {gemini_caption}")
        print(f"References: {references}")
        print(f"Hugging Face caption: {other_caption}")

        # Combine multiple captions into a single string
        gemini_caption = " ".join(gemini_caption) if isinstance(gemini_caption, list) else gemini_caption
        references = [" ".join(ref) if isinstance(ref, list) else ref for ref in references]

        # Embed captions
        gemini_embedding = embed_text(gemini_caption)
        reference_embeddings = [embed_text(ref) for ref in references]

        # Compute cosine similarity scores for Gemini captions
        gemini_scores = []
        for ref_embedding in reference_embeddings:
            if gemini_embedding is None or ref_embedding is None:
                print(f"Skipping invalid embedding for {filename}.")
                continue
            gemini_scores.append(cosine_similarity(gemini_embedding, ref_embedding))

        gemini_similarity_max = max(gemini_scores) if gemini_scores else 0
        gemini_similarity_avg = sum(gemini_scores) / len(gemini_scores) if gemini_scores else 0

        # Embed the other model's caption
        other_embedding = embed_text(other_caption)
        other_model_scores = []
        for ref_embedding in reference_embeddings:
            if other_embedding is None or ref_embedding is None:
                print(f"Skipping invalid embedding for {filename}.")
                continue
            other_model_scores.append(cosine_similarity(other_embedding, ref_embedding))

        other_model_similarity_max = max(other_model_scores) if other_model_scores else 0
        other_model_similarity_avg = sum(other_model_scores) / len(other_model_scores) if other_model_scores else 0

        # Append results
        results.append({
            "image": filename,
            "gemini_caption": gemini_caption,
            "other_model_caption": other_caption,
            "gemini_similarity_max": round(gemini_similarity_max, 4) if gemini_similarity_max else "N/A",
            "gemini_similarity_avg": round(gemini_similarity_avg, 4) if gemini_similarity_avg else "N/A",
            "other_model_similarity_max": round(other_model_similarity_max, 4) if other_model_similarity_max else "N/A",
            "other_model_similarity_avg": round(other_model_similarity_avg, 4) if other_model_similarity_avg else "N/A",
        })

    # Save results to CSV
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"Saved post-testing evaluation results to {output_csv}")