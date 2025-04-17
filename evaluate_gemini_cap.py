
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
from embedding_utils import cosine_similarity, embed_text
import numpy as np
import sqlite3
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

def evaluate_captions(image_dir, reference_path, output_csv, prompt):
    # Connect to DB and initialize model
    conn = init_db("labels.db")
    model = ModelApi()

    # Label any new images not in the DB
    label_images(image_dir, model, conn, prompt)

    # Load Gemini captions from DB
    gemini_captions_raw = get_all_labels(conn, prompt)
    gemini_captions = {
        os.path.basename(path): caption
        for path, caption in gemini_captions_raw.items()
    }

    # Load COCO reference captions
    with open(reference_path, "r") as f:
        reference_captions = json.load(f)

    """# Combine Gemini and reference captions into rows
    rows = []
    for image_name, gemini_caption in gemini_captions.items():
        reference_caption = reference_captions.get(image_name)
        if reference_caption:
            rows.append((image_name, gemini_caption, reference_caption))
"""
    results = []

    for md5, gemini_caption, reference_caption in rows:
        if not gemini_caption or not reference_caption:
            print(f"Skipping {md5}: Missing captions.")
            continue

        # Ensure inputs are strings
        if not isinstance(gemini_caption, str) or not isinstance(reference_caption, str):
            print(f"Invalid captions for {md5}: Gemini - {gemini_caption}, Reference - {reference_caption}")
            continue

        # Generate embeddings for captions
        gemini_embedding = embed_text(gemini_caption)
        reference_embedding = embed_text(reference_caption)

        if gemini_embedding is None or np.all(gemini_embedding == 0):
            print(f"Skipping invalid Gemini embedding for {md5}.")
            continue
        if reference_embedding is None or np.all(reference_embedding == 0):
            print(f"Skipping invalid Reference embedding for {md5}.")
            continue

        # Calculate similarity scores
        similarity_score = cosine_similarity(gemini_embedding, reference_embedding)

        results.append({
            "md5": md5,
            "gemini_caption": gemini_caption,
            "reference_caption": reference_caption,
            "similarity_score": round(similarity_score, 4)
        })

    # Save results to CSV
    if results:
        with open(output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"Saved evaluation results to {output_csv}")
    else:
        print("No valid results to save.")

if __name__ == "__main__":
    conn = sqlite3.connect("labels.db")
    evaluate_captions(
        conn=conn,
        output_csv="data/coco_subset/similarity_scores.csv"
    )
    conn.close()