import json
import csv
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

"""
Post-testing evaluation script.

Loads predicted captions and ground truth captions from JSON files.

Computes BLEU scores for each image and outputs a CSV with:
- BLEU-1, BLEU-2, BLEU-3, BLEU-4 scores
- Predicted caption
- Ground truth captions

Output saved in data/coco_subset/post_test_evaluation.csv
"""

def evaluate_post_testing(predicted_path, ground_truth_path, output_csv):
    # Load predicted captions
    with open(predicted_path, "r") as f:
        predicted_captions = json.load(f)

    # Load ground truth captions
    with open(ground_truth_path, "r") as f:
        ground_truth_captions = json.load(f)

    results = []
    smoothing_function = SmoothingFunction().method1

    for filename, predicted_caption in predicted_captions.items():
        references = ground_truth_captions.get(filename, [])
        if not references:
            continue

        # Compute BLEU scores
        bleu_1 = sentence_bleu(references, predicted_caption, weights=(1, 0, 0, 0), smoothing_function=smoothing_function)
        bleu_2 = sentence_bleu(references, predicted_caption, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing_function)
        bleu_3 = sentence_bleu(references, predicted_caption, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothing_function)
        bleu_4 = sentence_bleu(references, predicted_caption, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing_function)

        results.append({
            "image": filename,
            "bleu_1": round(bleu_1, 4),
            "bleu_2": round(bleu_2, 4),
            "bleu_3": round(bleu_3, 4),
            "bleu_4": round(bleu_4, 4),
            "predicted_caption": predicted_caption,
            "ground_truth_captions": references
        })

    # Save results to CSV
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"Saved post-testing evaluation results to {output_csv}")


if __name__ == "__main__":
    # Paths for post-testing evaluation
    predicted_path = "data/coco_subset/predicted_captions.json"
    ground_truth_path = "data/coco_subset/references.json"
    output_csv = "data/coco_subset/post_test_evaluation.csv"

    evaluate_post_testing(predicted_path, ground_truth_path, output_csv)