from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from PIL import Image
import os
import json
from db import init_db, label_images

def generate_captions_with_vit_gpt2(image_dir, db_path="labels.db"):
    """
    Generate captions using the ViT-GPT2 model and store them in a database.
    """
    # Load the pre-trained ViT-GPT2 model and tokenizer
    model = VisionEncoderDecoderModel.from_pretrained("ydshieh/vit-gpt2-coco-en")
    processor = ViTImageProcessor.from_pretrained("ydshieh/vit-gpt2-coco-en")
    tokenizer = AutoTokenizer.from_pretrained("ydshieh/vit-gpt2-coco-en")

    # Initialize the database
    conn = init_db(db_path)

    for filename in os.listdir(image_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(image_dir, filename)

            # Check if the image is already labeled
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

                # Store the caption in the database
                cursor.execute(
                    "INSERT INTO images (md5, image_path, label) VALUES (?, ?, ?)",
                    (file_hash, image_path, caption)
                )
                conn.commit()
                print(f"Generated caption for {filename}: {caption}")
            else:
                print(f"Caption already exists for {filename}: {result[0]}")

    conn.close()


if __name__ == "__main__":
    # Default paths
    image_dir = "images"  # Path to the folder containing images

    # Ensure the input directory exists
    if not os.path.exists(image_dir):
        print(f"Error: The directory '{image_dir}' does not exist.")
    else:
        # Generate captions
        generate_captions_with_vit_gpt2(image_dir)