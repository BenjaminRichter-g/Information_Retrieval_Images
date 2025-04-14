import os
from gemini_api import ModelApi
from db import init_db, label_images
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import json
import time

def calculate_precision_recall(generated_caption, reference_captions):
    """
    Calculate precision, recall, and F1-score for a generated caption against reference captions.
    """
    generated_tokens = set(generated_caption.lower().split())
    reference_tokens = set(" ".join(reference_captions).lower().split())

    # Precision: Fraction of generated tokens that are in the reference
    precision = len(generated_tokens & reference_tokens) / len(generated_tokens) if generated_tokens else 0

    # Recall: Fraction of reference tokens that are in the generated caption
    recall = len(generated_tokens & reference_tokens) / len(reference_tokens) if reference_tokens else 0

    # F1-Score: Harmonic mean of precision and recall
    f1_score = (2 * precision * recall) / (precision + recall) if precision + recall > 0 else 0

    return precision, recall, f1_score

def caption_images_in_directory(image_dir, prompt, reference_captions_path="data/coco_subset/references.json"):
    model = ModelApi()
    conn = init_db("labels.db")

    # Load reference captions
    with open(reference_captions_path, "r") as f:
        reference_captions = json.load(f)

    # Generate captions and calculate metrics
    for filename in os.listdir(image_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            retry_count = 3
            for attempt in range(retry_count):
                try:
                    # Generate caption
                    generated_caption = model.imageQuery(os.path.join(image_dir, filename), prompt)
                    if generated_caption is None:
                        raise ValueError("Generated caption is None")

                    # Get reference captions for the image
                    references = reference_captions.get(filename, [])

                    # Calculate precision, recall, and F1-score
                    precision, recall, f1_score = calculate_precision_recall(generated_caption, references)

                    print(f"Image: {filename}")
                    print(f"Generated Caption: {generated_caption}")
                    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1_score:.4f}")

                    # Add a delay to avoid rate-limiting
                    print(f"Sleeping for {5} seconds to avoid rate-limiting...")
                    time.sleep(5)
                    break  # Exit retry loop if successful
                except Exception as e:
                    print(f"Attempt {attempt + 1}/{retry_count} failed for {filename}: {e}")
                    if "RESOURCE_EXHAUSTED" in str(e):
                        print("Rate limit exceeded. Sleeping for 10 seconds...")
                        time.sleep(10)  # Wait longer before retrying
                    else:
                        break  # Exit retry loop for non-rate-limiting errors
                    
    
if __name__ == "__main__":
    image_dir = "data/coco_subset/images"
    prompt = "Write a COCO-style caption for this image."
    caption_images_in_directory(image_dir, prompt)