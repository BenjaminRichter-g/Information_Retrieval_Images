import os
from hashlib import md5
import sqlite3
import time

def generate_captions(directory, model, conn, prompt):
    """Generates captions for images and stores them in the database."""
    cursor = conn.cursor()

    for filename in os.listdir(directory):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.heic')):
            full_path = os.path.join(directory, filename)

            # Compute the MD5 hash of the image
            with open(full_path, 'rb') as f:
                file_data = f.read()
                file_hash = md5(file_data).hexdigest()

            # Check if captions already exist
            cursor.execute("SELECT md5 FROM captions WHERE md5 = ?", (file_hash,))
            if cursor.fetchone():
                print(f"Captions already exist for {filename}. Skipping.")
                continue

            # Generate captions
            gemini_caption = model.imageQuery(full_path, prompt)
            huggingface_caption = model.huggingfaceQuery(full_path, prompt)
            time.sleep(4)  # Delay to stay within API limits

            if gemini_caption and huggingface_caption:
                # Save captions to the database
                cursor.execute("""
                    INSERT INTO captions (md5, gemini_caption, huggingface_caption)
                    VALUES (?, ?, ?)
                """, (file_hash, gemini_caption, huggingface_caption))
                conn.commit()
                print(f"Generated captions for {filename}: Gemini - {gemini_caption}, Hugging Face - {huggingface_caption}")
            else:
                print(f"Failed to generate captions for {filename}.")