

"""
Loads  COCO captions from JSON file and Gemini captions from db

if image is not in db, label it with gemini and add to db

Computes cosine similarity for each image

Outputs a CSV with max similarity, average similarity, gemini caption

Output saved in data/coco_subset/similarity_scores.csv
"""
import os
import json
import csv
from embedding_utils import cosine_similarity
from db import init_db, get_all_labels, label_images
from gemini_api import ModelApi


prompts = [
    "Describe what is happening in this image.",
    "List the main objects visible in this image.",
    "Write a short sentence about the scene in this image.",
    "Write a COCO-style caption for this image.",
    "What are the people or animals doing in this image?",
    "Generate a short, realistic caption like those in the MS-COCO dataset.",
    "Write a caption that describes the main activity in this photo.",
]

def evaluate_captions(image_dir, reference_path, output_csv,prompt):
    # Connect to DB and initialize model
    conn = init_db("labels.db")
    model = ModelApi()

    # Label any new images not in the DB
    label_images(image_dir, model, conn,prompt)

    # Load Gemini captions from DB
    gemini_captions_raw = get_all_labels(conn,prompt)
    gemini_captions = {
        os.path.basename(path): caption
        for path, caption in gemini_captions_raw.items()
    }

    # Load COCO reference captions
    with open(reference_path, "r") as f:
        reference_captions = json.load(f)

    results = []
    total_max = 0
    total_avg = 0
    count = 0

    for filename, gemini_caption in gemini_captions.items():
        references = reference_captions.get(filename, [])
        if not references:
            continue

        scores = [cosine_similarity(gemini_caption, ref) for ref in references]
        max_score = max(scores)
        avg_score = sum(scores) / len(scores)

        total_max += max_score
        total_avg += avg_score
        count += 1

        results.append({
            "image": filename,
            "similarity_min": round(min(scores), 4),
            "similarity_max": round(max_score, 4),
            "similarity_avg": round(avg_score, 4),
            "gemini_caption": gemini_caption
        })

    # Save results
    avg_max_score = round(total_max / count, 4)
    avg_avg_score = round(total_avg / count, 4)
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
        writer.writerow({
            "image": "AVERAGE",
            "similarity_min": "",
            "similarity_max": avg_max_score,
            "similarity_avg": avg_avg_score,
            "gemini_caption": "N/A"
        })

    print(f"✅ Saved evaluation results to {output_csv}")
    return avg_max_score,avg_avg_score


if __name__ == "__main__":
    prompts = [
    "Describe what is happening in this image.",
    "List the main objects visible in this image.",
    "Write a short sentence about the scene in this image.",
    "Write a COCO-style caption for this image.",
    "What are the people or animals doing in this image?",
    "Generate a short, realistic caption like those in the MS-COCO dataset.",
    "Write a caption that describes the main activity in this photo.",
    ]
    image_dir="data/coco_subset/images"
    reference_path="data/coco_subset/references.json"
    #output_csv="data/coco_subset/similarity_scores.csv"
    summary_csv = "data/prompt_scores.csv"

    summary_results = []
       
    #evaluate_captions()
    for i, prompt in enumerate(prompts):
        print(f"\n🔍 Evaluating prompt {i+1}/{len(prompts)}: {prompt}\n")
        out_csv = f"data/coco_subset/similarity_scores_prompt_{i}.csv"
        avg_max, avg_avg = evaluate_captions(
            image_dir=image_dir,
            reference_path=reference_path,
            output_csv=out_csv,
            prompt=prompt
        )
        summary_results.append({
            "prompt": prompt,
            "average_max_similarity": avg_max,
            "average_avg_similarity": avg_avg
        })

    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["prompt", "average_max_similarity", "average_avg_similarity"])
        writer.writeheader()
        writer.writerows(summary_results)

    print(f"\n📄 Prompt-level summary saved to {summary_csv}")