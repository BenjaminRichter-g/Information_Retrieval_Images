import json
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from embedding_utils import cosine_similarity

"""
Compares generated captions to ground truth captions.

Calculates BLEU scores and cosine similarity for each image.

Outputs a JSON file with detailed comparison results.
"""

def compare_captions(generated_path, ground_truth_path, output_path):
    # Load generated captions
    with open(generated_path, "r") as f:
        generated_captions = json.load(f)

    # Load ground truth captions
    with open(ground_truth_path, "r") as f:
        ground_truth_captions = json.load(f)

    results = []
    smoothing_function = SmoothingFunction().method1

    for filename, generated_caption in generated_captions.items():
        references = ground_truth_captions.get(filename, [])
        if not references:
            continue

        # Calculate BLEU scores
        bleu_1 = sentence_bleu(references, generated_caption, weights=(1, 0, 0, 0), smoothing_function=smoothing_function)
        bleu_2 = sentence_bleu(references, generated_caption, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing_function)
        bleu_3 = sentence_bleu(references, generated_caption, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothing_function)
        bleu_4 = sentence_bleu(references, generated_caption, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing_function)

        # Calculate cosine similarity
        cosine_scores = [cosine_similarity(generated_caption, ref) for ref in references]
        max_cosine = max(cosine_scores) if cosine_scores else 0

        results.append({
            "image": filename,
            "generated_caption": generated_caption,
            "ground_truth_captions": references,
            "bleu_1": round(bleu_1, 4),
            "bleu_2": round(bleu_2, 4),
            "bleu_3": round(bleu_3, 4),
            "bleu_4": round(bleu_4, 4),
            "max_cosine_similarity": round(max_cosine, 4)
        })

    # Save results to JSON
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved comparison results to {output_path}")


if __name__ == "__main__":
    generated_path = "data/coco_subset/post_test_generated_captions.json"
    ground_truth_path = "data/coco_subset/references.json"
    output_path = "data/coco_subset/post_test_comparison.json"

    compare_captions(generated_path, ground_truth_path, output_path)