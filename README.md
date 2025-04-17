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


5. You'll need to have docker installed for the next part, as we're going to install Milvus. We couldn't load the docker folder containing the data in the 
project so PLEASE DOWNLOAD THE DATA from the following link and put it in the root directory:

```
https://drive.google.com/drive/folders/1G0zSrczbMgX7cY1I2PFeF0bbqSCSoPis?usp=drive_link
```

This is our vectorised Milvus database and is required to run the code.
Make sure Docker is opened.
Once the docker folder is put in the root directory, navigate inside and execute:
There is a README in there but all you need to do is:
```
bash standalone_embed.sh start 
```
OR for WINDOWS
```
standalone.bat start
```
to install and launch the milvus server

6. Two modes of use, you can use: (This is not recommended as its a lengthy procedure, the --small-test described bellow allows you to experience a small subset of the process)

    1. --create-label // in order to label all the images in the images folder with gemini 
    2. --embed-text // in order to embed the created descriptions

or you can use both to accomplish both activities at once, you can also use:
```
python main.py --help
```
for a quick description of the options

7. To execute the small test simply run:
```
python main.py --small-test
```
and it'll show you the process of creating a caption and an embedding for 3 randomly selected images.

## Running the app to use like in demo (Step 1 to 5 is required to run)

To run the app:

navigate to the directory which has server.py in and in the command line execute:
```
uvicorn server:app --reload
```

then navigate into the frontend directory and install everything and then run it:
```
npm install
npm start
```

# Gemini captioning and evaluation
This part of the project focuses on generating image captions using Google's Gemini model and evaluating them against COCO ground-truth captions using semantic similarity.

Tools Used
Gemini API (google-generativeai) for:
Caption generation
Text embedding (Gemini's embedding-001)
COCO dataset
Cosine similarity for semantic evaluation

Data used:
2017 Validation Images
Download from http://images.cocodataset.org/zips/val2017.zip (but already included in the repo)

2017 Captions Annotations
Download from http://images.cocodataset.org/annotations/annotations_trainval2017.zip


## Pre-Processing tests

To run the test comparing the coco and gemini caption generation use:
1. Generate a COCO subset:
```
python main.py --sample-coco
```

will create:
data/coco_subset/
├── images/
├── references.json
├── gemini_captions.json (after generation)
├── similarity_scores.csv (after evaluation)

2. Generate Gemini captions:
```
python caption_generator.py
```

3. Run evaluation:
```
python evaluate_gemini_cap.py
```

## Post-processing tests

To run the post-processing test, firs run: (This is not recommended as you will hit quota limits, the data is already in labels_raghav.db and the next command can be executed)
```
python main.py --create-label-tests
```
then run
```
python main.py --post-test
```
