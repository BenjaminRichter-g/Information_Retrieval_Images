import argparse
import gemini_api as ga
import embeddings as emb
from db import init_db, label_images, retrieve_images, drop_database
import vector_db as vd
from caption_generator_post import generate_captions_with_vit_gpt2  # Import for post-testing
from post_test_score import evaluate_post_testing  # Import for post-testing

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
        "--dir",
        type=str,
        default="images/",
        help="Directory containing images for label creation or post-testing (default: images/)."
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Completely resets the SQL AND MILVUS database!"
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

        for index in range(len(images)):
            milvus_db.insert_record(images[index][0], images[index][1], images[index][2], embedding[index][0].values)
        print("Inserted into Milvus done.")

        # Testing retrieval
        res = milvus_db.get_all_md5_hashes()
        print(res)

    if args.post_test:
        # Paths for post-testing
        image_dir = args.dir
        gemini_captions_path = "data/coco_subset/gemini_captions.json"
        other_model_captions_path = "data/coco_subset/other_model_captions.json"
        reference_captions_path = "data/coco_subset/references.json"
        output_csv_path = "data/coco_subset/post_test_similarity_scores.csv"

        # Step 1: Generate captions using the ViT-GPT2 model
        print("Generating captions using the ViT-GPT2 model...")
        generate_captions_with_vit_gpt2(image_dir, other_model_captions_path)

        # Step 2: Evaluate and compare captions
        print("Evaluating and comparing captions...")
        evaluate_post_testing(
            gemini_path=gemini_captions_path,
            other_model_path=other_model_captions_path,
            reference_path=reference_captions_path,
            output_csv=output_csv_path
        )
        print(f"Post-testing evaluation completed. Results saved to {output_csv_path}.")

    if args.reset:
        confirmation = input("Are you sure? This is not reversible and it might take a while to relabel and embed the images? Confirm with YES, anything else will be considered a no\n")
        if confirmation.lower() == "yes":
            drop_database()
            milvus_db = vd.MilvusDb()
            milvus_db.delete_entire_db()

    else:
        print("No valid operation specified. Use --create-label, --embed-text, or --post-test.")

if __name__ == "__main__":
    main()