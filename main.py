import argparse
import gemini_api as ga
import embeddings as emb
from db import init_db, label_images, retrieve_images, drop_database
import vector_db as vd
from caption_generator_post import generate_and_store_embeddings  # Import for post-testing
from post_test_score import evaluate_post_testing  # Import for post-testing
from search_embeddings import search_top_k_embeddings
import numpy as np

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
        "--sample-coco",
        action="store_true",
        help="Download and sample a subset of COCO images and captions."
    )
    parser.add_argument(
        "--dir",
        type=str,
        default="data/coco_subset/images",
        help="Directory containing images for label creation or post-testing (default: images/)."
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
    
    args = parser.parse_args()

    if args.create_label:
        model = ga.ModelApi()
        conn = init_db()
        label_images(args.dir, model, conn, prompt="Write a short sentence about the scene in this image.")
        conn.close()
        print("Label creation completed.")
    
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

        # Extract descriptions from the images
        descriptions = [image[2][:-1] for image in images]  # Adjusted to match the tuple structure
        embedding = embedder.batch_embeddings(descriptions)

        for index in range(len(images)):
            if index >= len(embedding) or embedding[index] is None:
                print(f"Skipping image {images[index][1]} due to missing or invalid embedding.")
                continue

            # Ensure the embedding is a flat list of floats
            embedding_vector = embedding[index]
            if isinstance(embedding_vector, np.ndarray):
                embedding_vector = embedding_vector.tolist()  # Convert NumPy array to list
            elif not isinstance(embedding_vector, list):
                print(f"Invalid embedding format for image {images[index][1]}. Skipping...")
                continue

            print(f"Inserting record for image: {images[index][1]}")
            milvus_db.insert_record(images[index][0], images[index][1], images[index][2], embedding_vector)        # Testing retrieval
        res = milvus_db.get_all_md5_hashes()
        print(res)

    if args.post_test:
    # Paths for post-testing
        image_dir = args.dir
        gemini_captions_path = "data/coco_subset/references.json"
        other_model_captions_path = "data/coco_subset/other_model_captions.json"
        reference_captions_path = "data/coco_subset/references.json"
        output_csv_path = "data/coco_subset/post_test_similarity_scores.csv"

        # Step 1: Generate captions using the ViT-GPT2 model
        print("Generating captions using the ViT-GPT2 model...")
        generate_and_store_embeddings(image_dir, db_path='labels.db', output_path=other_model_captions_path)

        # Step 2: Evaluate and compare captions
        print("Evaluating and comparing captions...")
        evaluate_post_testing(
            gemini_path=gemini_captions_path,
            other_model_path=other_model_captions_path,
            reference_path=reference_captions_path,
            output_csv=output_csv_path
        )
        print(f"Post-testing evaluation completed. Results saved to {output_csv_path}.")
        
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
            milvus_db = vd.MilvusDb()
            milvus_db.delete_entire_db()

    if args.search:
        query_prompt = input("Enter your query prompt: ")
        top_k_results = search_top_k_embeddings(query_prompt, k=10)

        print("\nTop-10 Similar Embeddings:")
        for result in top_k_results:
            print(f"MD5: {result['md5']}, Caption: {result['caption']}, Score: {result['score']}")

    else:
        print("No valid operation specified. Use --create-label, --embed-text, --post-test, or --search.")

if __name__ == "__main__":
    main()