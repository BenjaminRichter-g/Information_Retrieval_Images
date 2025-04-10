
import json
import os
import random
import shutil
import torchvision
from torchvision.datasets import CocoCaptions

"""
loads COCO captions via file thats dwnloaded manually

samples a subsest of image-caption pairs

Saves selected images to data/coco_subset/images/

Saves corresponding COCO captions to data/coco_subset/references.json
"""



def load_coco_dataset(
    root_dir="data/coco/images/val2017",
    ann_file="data/coco/annotations/captions_val2017.json"
):
    dataset = CocoCaptions(root=root_dir, annFile=ann_file)
    print(f"Loaded {len(dataset)} samples from COCO.")
    return dataset

def sample_coco_subset(dataset, num_samples=100):
    indices = random.sample(range(len(dataset)), num_samples)
    samples = []
    for idx in indices:
        image, captions = dataset[idx]
        image_id = dataset.ids[idx]
        samples.append({
            "image_id": image_id,
            "image": image,
            "captions": captions
        })
    return samples


def save_coco_subset(samples, output_dir="data/coco_subset"):
    os.makedirs(f"{output_dir}/images", exist_ok=True)

    references = {}

    for sample in samples:
        image_id = sample["image_id"]
        filename = f"{image_id:012d}.jpg"
        filepath = os.path.join(output_dir, "images", filename)
        sample["image"].save(filepath)

        references[filename] = sample["captions"]

    with open(os.path.join(output_dir, "references.json"), "w") as f:
        json.dump(references, f, indent=2)

    print(f"Saved {len(samples)} images and captions to {output_dir}.")