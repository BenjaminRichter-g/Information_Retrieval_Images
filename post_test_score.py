import json
import csv

"""
Aggregates scores from the comparison results.

Outputs a CSV file with average BLEU scores and cosine similarity.
"""

def aggregate_scores(comparison_path, output_csv):
    with open(comparison_path, "r") as f:
        comparisons = json.load(f)

    total_bleu_1 = total_bleu_2 = total_bleu_3 = total_bleu_4 = total_cosine = 0
    count = len(comparisons)

    for result in comparisons:
        total_bleu_1 += result["bleu_1"]
        total_bleu_2 += result["bleu_2"]
        total_bleu_3 += result["bleu_3"]
        total_bleu_4 += result["bleu_4"]
        total_cosine += result["max_cosine_similarity"]

    # Calculate averages
    avg_bleu_1 = total_bleu_1 / count
    avg_bleu_2 = total_bleu_2 / count
    avg_bleu_3 = total_bleu_3 / count
    avg_bleu_4 = total_bleu_4 / count
    avg_cosine = total_cosine / count

    # Save results to CSV
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Score"])
        writer.writerow(["BLEU-1", round(avg_bleu_1, 4)])
        writer.writerow(["BLEU-2", round(avg_bleu_2, 4)])
        writer.writerow(["BLEU-3", round(avg_bleu_3, 4)])
        writer.writerow(["BLEU-4", round(avg_bleu_4, 4)])
        writer.writerow(["Cosine Similarity", round(avg_cosine, 4)])

    print(f"Saved aggregated scores to {output_csv}")


if __name__ == "__main__":
    comparison_path = "data/coco_subset/post_test_comparison.json"
    output_csv = "data/coco_subset/post_test_scores.csv"

    aggregate_scores(comparison_path, output_csv)