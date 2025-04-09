import os
from gemini_api import ModelApi
from db import init_db, label_images

"""
Scans images in data/coco_subset/images/

Uses ModelApi.imageQuery() to generate captions
Stores them in labels.db (if not already present)
"""

def caption_images_in_directory(image_dir):
    model = ModelApi()
    conn = init_db("labels.db")
    label_images(image_dir, model, conn)

if __name__ == "__main__":
    image_dir = "data/coco_subset/images"
    caption_images_in_directory(image_dir)