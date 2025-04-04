import os
import json
from gemini_api import ModelApi
"""
Loops through all images in data/coco_subset/images/

Uses MOdelApi.imageQuesry() to generate gemini captions

Saves generated captions to data/coco_subset/gemini_captions.json
"""

def generate_captions_for_folder(image_dir, prompt="Describe the image."):
    model = ModelApi()
    captions = {}

    for filename in os.listdir(image_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(image_dir, filename)
            print(f"Processing {filename}...")
            caption = model.imageQuery(image_path, prompt)
            if caption:
                captions[filename] = caption

    return captions


def save_captions_to_json(captions, output_path):
    with open(output_path, "w") as f:
        json.dump(captions, f, indent=2)
    print(f"Saved {len(captions)} captions to {output_path}.")


if __name__ == "__main__":
    # For standalone test
    image_dir = "data/coco_subset/images"
    output_json = "data/coco_subset/gemini_captions.json"

    captions = generate_captions_for_folder(image_dir)
    save_captions_to_json(captions, output_json)