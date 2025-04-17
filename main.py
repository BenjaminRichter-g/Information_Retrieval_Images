import argparse
from PIL.Image import init
import gemini_api as ga
import embeddings as emb
from db import *
import vector_db as vd
from post_test_score import *
import numpy as np
import time

def main():
    parser = argparse.ArgumentParser(
        description="Run label creation, text embedding, or post-testing operations."
    )
    parser.add_argument(
        "--create-label",
        action="store_true",
        help="Create labels for images in a directory."
    )
    parser.add_argument(
        "--create-label-tests",
        action="store_true",
        help="Create labels for test images in a directory."
    )
    parser.add_argument(
        "--embed-text",
        action="store_true",
        help="Text to embed using the Google Text Embedder API."
    )
    parser.add_argument(
        "--post-test",
        action="store_true",
        help="Run post-testing evaluation (compare Gemini and other model captions)."
    )
    parser.add_argument(
        "--sample-coco",
        action="store_true",
        help="Download and sample a subset of COCO images and captions."
    )
    parser.add_argument(
        "--dir",
        type=str,
        default="data/coco_validation_2017/val2017",
        help="Directory containing images for label creation (default: images/)."
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Completely resets the SQL AND MILVUS database!"
    )
    parser.add_argument(
        "--search",
        action="store_true",
        help="Search for the top-k most similar embeddings based on a query prompt."
    )

    parser.add_argument(
        "--migrate-db",
        action="store_true",
        help="Run database migration to ensure the schema is up to date."
    )


    parser.add_argument(
            "--show-db",
            action="store_true",
            help = "uniquely for testing, shows what the db contains"
    )  

    parser.add_argument(
            "--small-test",
            action="store_true",
            help = "Small test to see how the captioning and embedding process works, note this generates captions and embeddings but does not save them. An API key is required"
    )

    args = parser.parse_args()

  
    prompt = "Generate a short, realistic caption like those in the MS-COCO dataset."
    
    if args.create_label:
        print("Starting the labeling process...")
        model = ga.ModelApi()
        conn = init_db()
        label_images(args.dir, model, conn, prompt)
        conn.close()
        print("Label creation completed.")

    if args.small_test:
        model = ga.ModelApi()
        conn = init_db("small_test.db")
        label_images("data/test_subset", model, conn, prompt)

        embedder = emb.Embedder()
        images = retrieve_all_images(conn)

        nb_images = len(images)
        nb_done = 0
        for index in range(len(images)):
            print(f"Finished the {nb_done} out of {nb_images}")
            print(f"Embeded caption: {images[index][3]}")
            nb_done+=1
            try:
                time.sleep(4)
                embedding = embedder.get_embedding(images[index][3])
            except Exception as e:
                time.sleep(1)
                print(e)


    if args.create_label_tests:
        print("Starting the labeling process...")
        model = ga.ModelApi(init_hf=True)
        conn = init_db()
        label_images_tests('data/tests', model, conn, prompt="Generate a short, realistic caption like those in the MS-COCO dataset.")
        embedder = emb.Embedder()
        captions = retrieve_captions(conn)

        clean_embeddings_test(conn)

        nb_captions = len(captions)
        nb_done = 0
        for caption in captions:
            print(f"Processing caption {nb_done}/{nb_captions}...")
            nb_done += 1
            time.sleep(10)
            try:
                if caption[1] is None or caption[2] is None:
                    print(f"Skipping caption {caption[0]} due to missing content.")
                    continue

                print(f"Generating embedding for content: {caption[1]}, {caption[2]}")
                
                try:
                    gemini_embed, hf_embed = embedder.double_embedding_test(caption[1], caption[2])
                    if gemini_embed is None or hf_embed is None:
                        print(f"Invalid embeddings for caption {caption[0]}. Skipping...")
                        continue

                    save_embedding(conn, caption[0], gemini_embed, hf_embed)
                except Exception as e:
                    print(f"Error processing caption {caption[0]}: {e}")
                    continue
                
                if gemini_embed is None or hf_embed is None:
                    print(f"Invalid embeddings for caption {caption[0]}. Skipping...")
                    continue

                if isinstance(gemini_embed, np.ndarray):
                    print(f"Gemini embedding shape: {gemini_embed.shape}")
                else:
                    print("Gemini embedding is invalid.")

                if isinstance(hf_embed, np.ndarray):
                    print(f"Hugging Face embedding shape: {hf_embed.shape}")
                else:
                    print("Hugging Face embedding is invalid.")

                save_embedding(conn, caption[0], gemini_embed, hf_embed)
            except Exception as e:
                print(f"Error processing caption {caption[0]}: {e}")
                time.sleep(2)

        # Retrieve and inspect embeddings
        try:
            embeddings = retrieve_embeddings(conn)
            for gemini_embedding, huggingface_embedding in embeddings:
                print(f"Gemini embedding shape: {gemini_embedding.shape}")
                print(f"Hugging Face embedding shape: {huggingface_embedding.shape}")
        except Exception as e:
            print(f"Error retrieving embeddings: {e}")

        conn.close()
        print("Label creation completed.")    

    if args.sample_coco:
        from coco_utils import load_coco_dataset, sample_coco_subset,save_coco_subset
        dataset = load_coco_dataset()
        samples = sample_coco_subset(dataset,num_samples=100)
        save_coco_subset(samples)
        return

    if args.show_db:
        conn = init_db()
        print(len(retrieve_all_images(conn)))
        vector_db = vd.MilvusDb()
        print(len(vector_db.get_all_md5_hashes()))
        

    if args.embed_text:
        milvus_db = vd.MilvusDb()
        embedder = emb.Embedder()

        existing_hashes = milvus_db.get_all_md5_hashes()
        conn = init_db()

        try:
            images = retrieve_images(conn, existing_hashes)
            print(f"Retrieved images: {images}")  # Debug print
            conn.close()
        except Exception as e:
            print(e)
            conn.close()
            return


        nb_images = len(images)
        nb_done = 0
        for index in range(len(images)):
            print(f"Finished the {nb_done} out of {nb_images}")
            print(f"Embeded caption: {images[index][3]}")
            nb_done+=1
            try:
                time.sleep(4)
                embedding = embedder.get_embedding(images[index][3])
                if embedding is not None:
                  milvus_db.insert_record(images[index][0], images[index][1], images[index][2], embedding)
                else:
                  print("Skipped due to None return")
            except Exception as e:
                time.sleep(1)
                print(e)
        print("inserted into milvus done")


        # Testing retrieval
        res = milvus_db.get_all_md5_hashes()
        print(res)

    if args.post_test:
        conn = init_db("labels_raghav.db")

        embeddings = retrieve_embeddings(conn)

        evaluate_embedding_cosine_similarity(embeddings, output_csv="results/similarity_scores_embedding.csv")
        evaluate_top_n_similarity(embeddings, output_csv="results/similarity_scores_top_n.csv", top_n=10)
       
        
    if args.sample_coco:
        from coco_utils import load_coco_dataset, sample_coco_subset, save_coco_subset
        dataset = load_coco_dataset()
        samples = sample_coco_subset(dataset, num_samples=1000)
        save_coco_subset(samples)
        return

    if args.reset:
        confirmation = input("Are you sure? This is not reversible and it might take a while to relabel and embed the images? Confirm with YES, anything else will be considered a no\n")
        if confirmation.lower() == "yes":
            drop_database()


    else:
        print("No valid operation specified. Use --create-label, --embed-text, --post-test, or --search.")


if __name__ == "__main__":
    main()
