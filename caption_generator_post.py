from PIL import Image
import os
import json
from hashlib import md5
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from db import init_db, save_caption_and_metrics
from MAP import calculate_map
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer


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


def generate_captions(image_dir, output_path="data/coco_subset/other_model_captions.json", prompt="Write a COCO-style caption for this image.", reference_captions_path="data/coco_subset/references.json"):
    """
    Generate captions using the pre-trained ViT-GPT2 model and save them to a JSON file and database.
    """
    # Load the pre-trained ViT-GPT2 model and tokenizer
    model = VisionEncoderDecoderModel.from_pretrained("ydshieh/vit-gpt2-coco-en")
    processor = ViTImageProcessor.from_pretrained("ydshieh/vit-gpt2-coco-en")
    tokenizer = AutoTokenizer.from_pretrained("ydshieh/vit-gpt2-coco-en")

    # Initialize the database
    conn = init_db("labels.db")
    cursor = conn.cursor()

    # Load reference captions
    with open(reference_captions_path, "r") as f:
        reference_captions = json.load(f)

    # Load existing captions from the JSON file if it exists
    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            captions = json.load(f)
    else:
        captions = {}

    for filename in os.listdir(image_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(image_dir, filename)

            # Check if the image is already processed
            if filename in captions:
                print(f"Caption already exists for {filename}: {captions[filename]}")
                continue

            try:
                # Generate a caption for the image
                image = Image.open(image_path).convert("RGB")
                inputs = processor(images=image, return_tensors="pt")
                output_ids = model.generate(inputs.pixel_values)
                caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)

                # Calculate MD5 hash for the image file
                with open(image_path, "rb") as f:
                    md5_hash = md5(f.read()).hexdigest()

                # Get reference captions for the image
                references = reference_captions.get(filename, [])

                # Calculate precision, recall, and F1-score
                precision, recall, f1_score = calculate_precision_recall(caption, references)

                print(f"Image: {filename}")
                print(f"Generated Caption: {caption}")
                print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1_score:.4f}")

                # Add the caption to the JSON dictionary
                captions[filename] = caption

                # Save the results to the database
                save_caption_and_metrics(conn, md5_hash, image_path, caption, prompt, precision, recall, f1_score)
            except Exception as e:
                print(f"Failed to generate caption for {filename}: {e}")
                captions[filename] = "No caption generated"

    # Save captions to JSON
    with open(output_path, "w") as f:
        json.dump(captions, f, indent=2)
    print(f"Saved captions to {output_path}")

    # Calculate MAP
    map_score = calculate_map(captions, reference_captions)
    print(f"Mean Average Precision (MAP): {map_score:.4f}")

    conn.close()


if __name__ == "__main__":
    # Generate captions using the pre-trained model
    image_dir = "data/coco_subset/images"
    output_path = "data/coco_subset/ViT_GPT2_captions.json"
    prompt = "Write a COCO-style caption for this image."
    generate_captions(image_dir, output_path=output_path, prompt=prompt)