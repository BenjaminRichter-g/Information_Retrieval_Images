# server.py  :contentReference[oaicite:0]{index=0}
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles  # <-- NEW
from pydantic import BaseModel
from typing import List, Optional

import os

import gemini_api as ga
import embeddings as emb
from db import init_db, label_images, retrieve_images, drop_database
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


app.mount(
    "/data",  # URL prefix
    StaticFiles(directory="data"),  # local folder to serve
    name="data"
)
# ---------------------------------------------------------------------

# -------------------- Pydantic models --------------------
class LabelRequest(BaseModel):
    directory: str  # the folder in which images are stored

class EmbedRequest(BaseModel):
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
        # Provide a default prompt or pass one in if you like
        label_images(directory, model, conn, prompt="Describe what is in this image.")
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

        # Each row is (md5, file_path, prompt, label)
        images = retrieve_images(conn, existing_hashes)
        conn.close()

        # We'll embed the label field
        descriptions = [row[3] for row in images]

        # Batch embed
        embeddings = embedder.batch_embeddings(descriptions)

        for index in range(len(images)):
            md5_val = images[index][0]
            file_path = images[index][1]
            label = images[index][3]
            vector_values = embeddings[index][0].values  # pick out the float vector

            milvus_db.insert_record(
                md5_val,
                file_path,
                label,
                vector_values
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


@app.post("/search")
def search_endpoint(request: SearchRequest):
    """
    Searches the Milvus vector DB given a textual query by generating an embedding
    of the query and performing a vector similarity search.
    Returns image URLs that point to the static directory, so the frontend
    can render the images via <img src="...">.
    """
    # 1. embed the query
    embedder = emb.Embedder()
    try:
        query_embedding_result = embedder.get_embedding(request.query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    query_embedding = query_embedding_result[0].values  # single vector

    # 2. search in Milvus
    try:
        milvus_db = vd.MilvusDb()
        results = milvus_db.search_by_embedding(query_embedding, limit=request.limit)
        
        # Format results for the front-end
        output = []
        for result in results:
            for hit in result:
                local_path = hit.entity.get("file_path", "")
                if not local_path:
                    continue

                # Construct a serving URL
                # e.g. "http://localhost:8000/data/coco_validation_2017/val2017/..."
                image_url = f"http://localhost:8000/{local_path}"

                output.append({
                    "md5": hit.entity.get("md5", ""),
                    "file_path": image_url,  # <-- now a fully-qualified URL
                    "description": hit.entity.get("description", ""),
                    "distance": hit.distance
                })
        return {"results": output}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
