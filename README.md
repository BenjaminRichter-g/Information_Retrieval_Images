# Information Retrieval on image dataset

This is a repository consisting of the code required to both label an image dataset thanks to AI and 
then to search through this dataset through prompts.

This will eventually be usable with a pre-determined dataset but also serve as a software which can be fed
an arbitrary set of images, label them and allow you to search through them with prompts.

## How to Use

1. Get the Gemini api key

2. Create a .env file

3. Put this in your .env file

API_KEY=your_api_key  //btw you can get the APIE key straight from google ai studio, top left should be a button telling you get my api key

4. Do the steps bellow and it'll work //This also assumes you're working from terminal and create a virtual env

```
python3 -m venv InformationRetrievalEnv

source InformationRetrievalEnv/bin/activate

pip install -r requirements.txt
```


5. You'll need to have docker installed for the next part, as we're going to install Milvus. Go to the /docker folder. There is a README in there but all you need to do is:

```
bash standalone_embed.sh start 
```
to install and launch the milvus server


6. Two modes of use, you can use:

    1. --create-label // in order to label all the images in the images folder with gemini 
    2. --embed-text // in order to embed the created descriptions

or you can use both to accomplish both activities at once, you can also use:
```
python main.py --help
```
for a quick description of the options

# Main branch

the main branch is protected


# Gemini captioning and evaluation
This part of the project focuses on generating image captions using Google's Gemini model and evaluating them against COCO ground-truth captions using semantic similarity.

Tools Used
Gemini API (google-generativeai) for:
Caption generation
Text embedding (Gemini's embedding-001)
COCO dataset
Cosine similarity for semantic evaluation

* The data/ folder is not included in this repository to keep it lightweight. To reproduce the results, you’ll need to manually download and prepare the necessary COCO data.

 Step 1: Download COCO 2017 Files
2017 Validation Images
Download from http://images.cocodataset.org/zips/val2017.zip

2017 Captions Annotations
Download from http://images.cocodataset.org/annotations/annotations_trainval2017.zip

Extract the files to structure th eproject like this:
Information_Retrieval_Images/
├── data/
│   └── coco/
│       ├── images/
│       │   └── val2017/                   # From val2017.zip
│       └── annotations/
│           └── captions_val2017.json      # From annotations_trainval2017.zip



output structure:
data/
└── coco_subset/
    ├── images/                  # Selected COCO images
    ├── references.json          # COCO captions
    ├── gemini_captions.json     # Generated Gemini captions
    └── similarity_scores.csv    # Evaluation results

To run:
1. Generate a COCO subset:
python main.py --sample-coco

will create:
data/coco_subset/
├── images/
├── references.json
├── gemini_captions.json (after generation)
├── similarity_scores.csv (after evaluation)

2. Generate Gemini captions:
python caption_generator.py

3. Run evaluation:
python evaluate_gemini_cap.py