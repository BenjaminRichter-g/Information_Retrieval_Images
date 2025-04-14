import json
import csv
from embedding_utils import cosine_similarity
#errro needs t be handled
"""
Loads Gemini and COCO captions from JSON files

Computes cosine similarity for each image

Outputs a CSV with max similarity, average similarity, gemini caption
 
Output saved in data/coco_subset/similarity_scores.csv
"""

def evaluate_captions(gemini_path, reference_path, output_csv):
    with open(gemini_path, "r") as f:
        gemini_captions = json.load(f)

    with open(reference_path, "r") as f:
        reference_captions = json.load(f)

    results = []

    for filename, gemini_caption in gemini_captions.items():
        references = reference_captions.get(filename, [])
        if not references:
            continue

        scores = [cosine_similarity(gemini_caption, ref) for ref in references]
        max_score = max(scores)
        avg_score = sum(scores) / len(scores)

        results.append({
            "image": filename,
            "similarity_max": round(max_score, 4),
            "similarity_avg": round(avg_score, 4),
            "gemini_caption": gemini_caption
        })

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"Saved evaluation results to {output_csv}")


if __name__ == "__main__":
    evaluate_captions(
        gemini_path="data/coco_subset/gemini_captions.json",
        reference_path="data/coco_subset/references.json",
        output_csv="data/coco_subset/similarity_scores.csv"
    )