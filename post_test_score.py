import json
import csv
from embedding_utils import cosine_similarity

def evaluate_post_testing(gemini_path, other_model_path, reference_path, output_csv):
    """
    Compare captions from Gemini and another model, compute cosine similarity metrics, and save results.
    """
    # Load captions
    with open(gemini_path, "r") as f:
        gemini_captions = json.load(f)

    with open(other_model_path, "r") as f:
        other_model_captions = json.load(f)

    with open(reference_path, "r") as f:
        reference_captions = json.load(f)

    results = []

    for filename, gemini_caption in gemini_captions.items():
        references = reference_captions.get(filename, [])
        other_caption = other_model_captions.get(filename, "")

        if not references or not other_caption:
            continue

        # Compute cosine similarity scores for Gemini captions
        gemini_scores = [cosine_similarity(gemini_caption, ref) for ref in references]
        gemini_similarity_max = max(gemini_scores)
        gemini_similarity_avg = sum(gemini_scores) / len(gemini_scores)

        # Compute cosine similarity scores for the other model's captions
        other_model_scores = [cosine_similarity(other_caption, ref) for ref in references]
        other_model_similarity_max = max(other_model_scores)
        other_model_similarity_avg = sum(other_model_scores) / len(other_model_scores)

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
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"Saved post-testing evaluation results to {output_csv}")


if __name__ == "__main__":
    gemini_path = "data/coco_subset/gemini_captions.json"
    other_model_path = "data/coco_subset/other_model_captions.json"
    reference_path = "data/coco_subset/references.json"
    output_csv = "data/coco_subset/post_test_similarity_scores.csv"

    evaluate_post_testing(gemini_path, other_model_path, reference_path, output_csv)