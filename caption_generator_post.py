from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from PIL import Image
import os
import json
from hashlib import md5
from db import init_db
from embeddings import Embedder
import vector_db as vd  # Milvus integration

def generate_and_store_embeddings(image_dir, db_path="labels.db", output_path="data/coco_subset/other_model_captions.json"):
    """
    Generate captions using the ViT-GPT2 model, embed them using Gemini, and store in Milvus and a JSON file.
    """
    # Load the pre-trained ViT-GPT2 model and tokenizer
    model = VisionEncoderDecoderModel.from_pretrained("ydshieh/vit-gpt2-coco-en")
    processor = ViTImageProcessor.from_pretrained("ydshieh/vit-gpt2-coco-en")
    tokenizer = AutoTokenizer.from_pretrained("ydshieh/vit-gpt2-coco-en")

    # Initialize the database and Milvus
    conn = init_db(db_path)
    embedder = Embedder()
    milvus_db = vd.MilvusDb()

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

            with open(image_path, "rb") as f:
                file_data = f.read()
                file_hash = md5(file_data).hexdigest()

            cursor = conn.cursor()
            cursor.execute("SELECT label FROM images WHERE md5 = ?", (file_hash,))
            result = cursor.fetchone()

            if result is None:
                # Generate a caption for the image
                image = Image.open(image_path).convert("RGB")
                inputs = processor(images=image, return_tensors="pt")
                output_ids = model.generate(inputs.pixel_values)
                caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)
                print(f"Generated caption for {filename}: {caption}")  # Debug print

                # Embed the caption using Gemini
                embedding = embedder.get_embedding(caption)

                # Store the caption and embedding in the database and Milvus
                cursor.execute(
                    "INSERT INTO images (md5, image_path, label) VALUES (?, ?, ?)",
                    (file_hash, image_path, caption)
                )
                conn.commit()

                milvus_db.insert_record(file_hash, image_path, caption, embedding)
                print(f"Processed and stored embedding for {filename}: {caption}")

                # Add the caption to the JSON dictionary
                captions[filename] = caption
            else:
                print(f"Caption already exists in the database for {filename}: {result[0]}")
                captions[filename] = result[0]  # Add existing caption to JSON

    # Save captions to JSON
    with open(output_path, "w") as f:
        json.dump(captions, f, indent=2)
    print(f"Saved captions to {output_path}")

    conn.close()

if __name__ == "__main__":
    # Default paths
    image_dir = "data/coco_subset/images"  # Path to the folder containing images
    output_path = "data/coco_subset/other_model_captions.json"  # Path to save the captions

    # Ensure the input directory exists
    if not os.path.exists(image_dir):
        print(f"Error: The directory '{image_dir}' does not exist.")
    else:
        # Generate captions and store embeddings
        generate_and_store_embeddings(image_dir, output_path=output_path)