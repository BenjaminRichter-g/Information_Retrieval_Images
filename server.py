from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from fastapi.responses import FileResponse
import os

import gemini_api as ga
import embeddings as emb
from db import init_db, label_images, retrieve_images, drop_database, get_description_by_md5
import vector_db as vd

# -------------------- FastAPI setup --------------------
app = FastAPI()

# Enable CORS
origins = [
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------- Pydantic models --------------------

class LabelRequest(BaseModel):
    directory: str  # the folder in which images are stored

class EmbedRequest(BaseModel):
    # optionally add fields you need from the front-end, e.g. which images, or an entire text?
    pass

class ResetRequest(BaseModel):
    confirm: str

class SearchRequest(BaseModel):
    query: str
    limit: Optional[int] = 10

# -------------------- Endpoints --------------------

@app.get("/")
def read_root():
    return {"message": "Welcome to the FastAPI backend!"}

@app.post("/label-images")
def label_images_endpoint(request: LabelRequest):
    """
    Labels all images in the specified directory that are not yet in the SQLite DB.
    Calls the Gemini model to get a description and saves them to the DB.
    """
    directory = request.directory
    if not os.path.isdir(directory):
        raise HTTPException(status_code=400, detail="Directory not found.")

    model = ga.ModelApi()
    conn = init_db()
    try:
        label_images(directory, model, conn)
    finally:
        conn.close()

    return {"message": f"Labeling images in {directory} completed."}


@app.post("/embed-text")
def embed_text_endpoint(request: EmbedRequest):
    """
    Embeds text for all images that are not currently in Milvus.
    """
    try:
        milvus_db = vd.MilvusDb() 
        embedder = emb.Embedder()

        existing_hashes = milvus_db.get_all_md5_hashes() 
        conn = init_db()

        images = retrieve_images(conn, existing_hashes)
        conn.close()

        # The "batch_embeddings" method is used for the images' textual descriptions
        descriptions = [description[:-1] for (_, _, description) in images] 
        # We embed each description
        embedding = embedder.batch_embeddings(descriptions)

        # Insert each embedding into Milvus
        for index in range(len(images)):
            milvus_db.insert_record(
                images[index][0],  # md5
                images[index][1],  # file_path
                images[index][2],  # description
                embedding[index][0].values  # the actual embedding vector
            )
        
        return {"message": "Insertion into Milvus done."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reset-db")
def reset_db_endpoint(request: ResetRequest):
    """
    Resets the entire DB (both SQLite and Milvus).
    Use with caution. Must pass {"confirm":"YES"} to proceed.
    """
    if request.confirm.upper() == "YES":
        drop_database()
        milvus_db = vd.MilvusDb()
        milvus_db.delete_entire_db()
        return {"message": "All databases have been reset."}
    else:
        return {"message": "Reset operation cancelled."}

from fastapi.responses import FileResponse
import os

@app.get("/get-image/{filename}")
def get_image(filename: str):
    path = os.path.join("data/coco_validation_2017/val2017", filename)
    if os.path.isfile(path):
        return FileResponse(path)
    return {"error": "File not found"}

@app.post("/search")
def search_endpoint(request: SearchRequest):
    """
    Searches the Milvus vector DB given a textual query by generating an embedding
    of the query and performing a vector similarity search.
    """
    # 1. embed the query
    embedder = emb.Embedder()
    conn = init_db()
    try:
        query_embedding_result = embedder.get_embedding(request.query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    query_embedding = query_embedding_result  # single vector

    # 2. search in Milvus
    try:
        milvus_db = vd.MilvusDb()
        results = milvus_db.search_by_embedding(query_embedding, limit=request.limit)
        # Format results for the front-end
        output = []
        md5s = [
            hit.entity.get("md5")
            for result in results  # each result is a Hits object
            for hit    in result            # each hit is a Hit object
        ]
        file_paths =  [
            hit.entity.get("file_path")
            for result in results  # each result is a Hits object
            for hit    in result            # each hit is a Hit object
        ]
        distances =  [
            hit.distance
            for result in results  # each result is a Hits object
            for hit    in result            # each hit is a Hit object
        ]
        descriptions = [get_description_by_md5(conn, md5) for md5 in md5s]

        for index in range(len(results)):
            
            output.append({
                "md5": md5s[index],
                "file_path": f"http://localhost:8000/get-image/{os.path.basename(file_paths[index])}",
                "description": descriptions[index],
                "distance": distances[index]
            })
            #print(f"Result {index}: {output[index]}")
        return {"results": output}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
