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

