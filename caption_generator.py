import os
import time
import json
from hashlib import md5
from PIL import Image
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from db import init_db, save_caption_and_metrics
from MAP import calculate_map


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
    """
    Generate captions for images in a directory and calculate precision, recall, and F1-score.
    """
    # Load the pre-trained model, processor, and tokenizer
    model = VisionEncoderDecoderModel.from_pretrained("ydshieh/vit-gpt2-coco-en")
    processor = ViTImageProcessor.from_pretrained("ydshieh/vit-gpt2-coco-en")
    tokenizer = AutoTokenizer.from_pretrained("ydshieh/vit-gpt2-coco-en")

    conn = init_db("labels.db")
    cursor = conn.cursor()

    # Load reference captions
    with open(reference_captions_path, "r") as f:
        reference_captions = json.load(f)

    generated_captions = {}

    # Generate captions and calculate metrics
    for filename in os.listdir(image_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(image_dir, filename)  # Get the full path to the image

            retry_count = 3
            for attempt in range(retry_count):
                try:
                    # Generate caption
                    image = Image.open(image_path).convert("RGB")
                    inputs = processor(images=image, return_tensors="pt")
                    output_ids = model.generate(inputs.pixel_values)
                    generated_caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)

                    # Calculate MD5 hash for the image file
                    with open(image_path, "rb") as f:
                        md5_hash = md5(f.read()).hexdigest()

                    # Get reference captions for the image
                    references = reference_captions.get(filename, [])

                    # Calculate precision, recall, and F1-score
                    precision, recall, f1_score = calculate_precision_recall(generated_caption, references)

                    print(f"Image: {filename}")
                    print(f"Generated Caption: {generated_caption}")
                    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1_score:.4f}")

                    # Save the results to the database
                    cursor.execute(
                        "INSERT INTO captions (filename, caption, precision, recall, f1_score) VALUES (?, ?, ?, ?, ?)",
                        (filename, generated_caption, precision, recall, f1_score)
                    )
                    conn.commit()

                    # Add the caption to the dictionary
                    generated_captions[filename] = generated_caption

                    # Add a delay to avoid rate-limiting
                    print(f"Sleeping for 10 seconds to avoid rate-limiting...")
                    time.sleep(10)
                    break  # Exit retry loop if successful
                except Exception as e:
                    print(f"Attempt {attempt + 1}/{retry_count} failed for {filename}: {e}")
                    if "RESOURCE_EXHAUSTED" in str(e):
                        print("Rate limit exceeded. Sleeping for 10 seconds...")
                        time.sleep(10)  # Wait longer before retrying
                    else:
                        break  # Exit retry loop for non-rate-limiting errors

    conn.close()
    return generated_captions


if __name__ == "__main__":
    # Generate captions using the pre-trained model
    image_dir = "data/coco_subset/images"
    prompt = "Write a COCO-style caption for this image."
    generated_captions = caption_images_in_directory(image_dir, prompt)

    # Load reference captions
    with open("data/coco_subset/references.json", "r") as f:
        reference_captions = json.load(f)

    # Calculate MAP
    map_score = calculate_map(generated_captions, reference_captions)
    print(f"Mean Average Precision (MAP): {map_score:.4f}")