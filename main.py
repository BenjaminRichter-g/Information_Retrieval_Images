
import argparse
import gemini_api as ga
import embeddings as emb
from db import init_db, label_images

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
    
    elif args.embed_text:
        embedder = emb.
        embedding = embedder.embed(args.embed_text)
        print("Embedding result:", embedding)
    
    else:
        print("No valid operation specified. Use --create-label or --embed-text.")

if __name__ == "__main__":
    main()

