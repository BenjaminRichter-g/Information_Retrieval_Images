import json
import csv
from embedding_utils import cosine_similarity, embed_text
import numpy as np
import sqlite3

def evaluate_captions(conn, output_csv):
    """Evaluates captions by comparing Gemini captions with reference captions."""
    cursor = conn.cursor()

    # Fetch all Gemini captions and their corresponding references
    cursor.execute("""
        SELECT images.md5, captions.gemini_caption, images.label
        FROM captions
        INNER JOIN images ON captions.md5 = images.md5
    """)
    rows = cursor.fetchall()

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