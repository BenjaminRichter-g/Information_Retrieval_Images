from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from PIL import Image
import os
import json

def generate_captions_with_vit_gpt2(image_dir, output_path):
    """
    Generate captions using the ViT-GPT2 model.
    """
    # Load the pre-trained ViT-GPT2 model and tokenizer
    model = VisionEncoderDecoderModel.from_pretrained("ydshieh/vit-gpt2-coco-en")
    processor = ViTImageProcessor.from_pretrained("ydshieh/vit-gpt2-coco-en")
    tokenizer = AutoTokenizer.from_pretrained("ydshieh/vit-gpt2-coco-en")

    captions = {}

    for filename in os.listdir(image_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(image_dir, filename)
            image = Image.open(image_path).convert("RGB")

            # Preprocess the image and generate a caption
            inputs = processor(images=image, return_tensors="pt")
            output_ids = model.generate(inputs.pixel_values)
            caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)

            captions[filename] = caption
            print(f"Generated caption for {filename}: {caption}")

    # Save captions to JSON
    with open(output_path, "w") as f:
        json.dump(captions, f, indent=2)
    print(f"Saved captions to {output_path}")


if __name__ == "__main__":
    # Default paths
    image_dir = "images"  # Path to the folder containing images
    output_path = "data/coco_subset/post_test_generated_captions.json"  # Path to save the captions

    # Ensure the input directory exists
    if not os.path.exists(image_dir):
        print(f"Error: The directory '{image_dir}' does not exist.")
    else:
        # Generate captions
        generate_captions_with_vit_gpt2(image_dir, output_path)