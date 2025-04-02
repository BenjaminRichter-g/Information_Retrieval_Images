import argparse
import gemini_api as ga
import embeddings as emb
from db import init_db, label_images, retrieve_images
import vector_db as vd

def main():
    parser = argparse.ArgumentParser(
        description="Run label creation or text embedding operations."
    )
    parser.add_argument(
        "--create-label",
        action="store_true",
        help="Create labels for images in a directory."
    )
    parser.add_argument(
        "--embed-text",
        action="store_true",
        help="Text to embed using the Google Text Embedder API."
    )
    parser.add_argument(
        "--dir",
        type=str,
        default="images/",
        help="Directory containing images for label creation (default: images/)."
    )
    
    args = parser.parse_args()

    if args.create_label:
        model = ga.ModelApi()
        conn = init_db()
        label_images(args.dir, model, conn)
        conn.close()
        print("Label creation completed.")
    
    if args.embed_text:

        milvus_db = vd.MilvusDb() 
        embedder = emb.Embedder()

        existing_hashes = milvus_db.get_all_md5_hashes() 
        conn = init_db() 

        try:
            images = retrieve_images(conn, existing_hashes)
            conn.close()
        except Exception as e:
            print(e)
            conn.close()
            return
        
        descriptions = [description[:-1] for (_, _, description) in images] 
        embedding = embedder.batch_embeddings(descriptions)

        print("Embedding result:", embedding)
        for index in range(len(images)):
            milvus_db.insert_record(images[index][0], images[index][1], images[index][2], embedding[index])
        print("inserted into milvus done")

        #testing retrieval

        res = milvus_db.get_all_md5_hashes()
        print(res)

    else:
        print("No valid operation specified. Use --create-label or --embed-text.")

if __name__ == "__main__":
    main()

