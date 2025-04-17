import sqlite3
import os
from hashlib import md5
import time
import numpy as np
import pickle

def init_db(db_path="labels_raghav.db"):
    """Initializes the SQLite database and creates the necessary tables if they don't exist."""
    db_path = os.path.abspath(db_path)  # Make path absolute
    print(f"🔍 Initializing DB at: {db_path}")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create the images table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS tests (
            md5 TEXT PRIMARY KEY,
            image_path TEXT NOT NULL,
            prompt TEXT NOT NULL
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            md5 TEXT,
            image_path TEXT NOT NULL,
            label TEXT,
            prompt TEXT NOT NULL,
            UNIQUE(md5)
        )
    """)

    # Create the captions table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS captions (
            md5 TEXT PRIMARY KEY,
            gemini_caption TEXT,
            huggingface_caption TEXT,
            FOREIGN KEY (md5) REFERENCES tests (md5)
        )
    """)

    # Create the embeddings table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS embeddings (
            md5 TEXT PRIMARY KEY,
            gemini_embedding BLOB,
            huggingface_embedding BLOB,
            FOREIGN KEY (md5) REFERENCES tests (md5)
        )
    """)

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS images (
            md5 TEXT,
            image_path TEXT,
            prompt TEXT,
            label TEXT,
            PRIMARY KEY (md5, prompt)
        )
    ''')

    conn.commit()
    return conn

def migrate_db(db_path="labels.db"):
    """ 
    Migrates the existing data to the new schema to prvent the relabeling of images
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create a new table with the updated schema
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS images_new (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            md5 TEXT,
            image_path TEXT,
            label TEXT,
            prompt TEXT,
            UNIQUE(md5, prompt)
        )
    """)

    # Copy data from the old table to the new table
    cursor.execute("""
        INSERT OR IGNORE INTO images_new (md5, image_path, label, prompt)
        SELECT md5, image_path, label, prompt FROM images
    """)

    # Drop the old table and rename the new table
    cursor.execute("DROP TABLE images")
    cursor.execute("ALTER TABLE images_new RENAME TO images")

    conn.commit()
    conn.close()
    print("Database migration completed.")

def label_images(directory, model, conn, prompt):
    """Iterates over image files in the given directory, labels those not already in the database, and stores the results."""
    cursor = conn.cursor()

    nb_files = len([name for name in os.listdir(directory)])
    files_processed = 0
    for filename in os.listdir(directory):
        print(f"Handling file {files_processed} out of {nb_files}")
        files_processed+=1
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.heic')):
            full_path = os.path.join(directory, filename)
            # Check if this image is already labeled with the given prompt
            cursor.execute("SELECT md5 FROM images WHERE image_path = ? AND prompt = ?", (full_path, prompt))

            if cursor.fetchone() is None:
                # Call the model to label the image
                description = model.imageQuery(full_path, prompt)
                if description:
                    with open(full_path, 'rb') as f:
                        file_data = f.read()
                        file_hash = md5(file_data).hexdigest()
                    #addd imahe with prompt
                    cursor.execute(
                    "INSERT INTO images (md5, image_path, prompt, label) VALUES (?, ?, ?, ?)",
                    (file_hash, full_path, prompt, description)
                    )

                    conn.commit()
                    print(f"Labeled {filename}: {description}")
                else:
                    print(f"Failed to label {filename}")
            else:
                print(f"Already labeled {filename}")

def label_images_tests(directory, model, conn, prompt):
    """Iterates over image files in the given directory, labels those not already in the database, and stores the results."""
    cursor = conn.cursor()
    total_images = len([f for f in os.listdir(directory) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.heic'))])
    labeled_count = 0
    already_labeled_count = 0

    for index, filename in enumerate(os.listdir(directory)):
        print(f"Processing {index + 1}/{total_images} images...")
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.heic')):
            full_path = os.path.join(directory, filename)

            # Compute the MD5 hash of the image
            print(full_path)
            with open(full_path, 'rb') as f:
                file_data = f.read()
                file_hash = md5(file_data).hexdigest()

            print(file_hash)

            # Check if the image is already labeled in the `tests` table
            cursor.execute("SELECT md5 FROM tests WHERE md5 = ?", (file_hash,))
            if cursor.fetchone() is not None:
                already_labeled_count += 1
                print(f"[{index + 1}/{total_images}] Already labeled {filename}")
                continue

            # Generate captions for the image
            description_gemini = model.imageQuery(full_path, prompt)
            description_hf = model.huggingfaceQuery(full_path)  # Removed 'prompt' argument
            time.sleep(4)  # 4s delay to stay within ~15 requests/min

            if description_gemini or description_hf:
                # Insert into the `tests` table
                cursor.execute("""
                    INSERT INTO tests (md5, image_path, prompt)
                    VALUES (?, ?, ?)
                """, (file_hash, full_path, prompt))

                # Insert into the `captions` table
                cursor.execute("""
                    INSERT INTO captions (md5, gemini_caption, huggingface_caption)
                    VALUES (?, ?, ?)
                """, (file_hash, description_gemini, description_hf))

                conn.commit()
                labeled_count += 1
                print(f"[{index + 1}/{total_images}] Labeled GG{filename}: {description_gemini}")
                print(f"[{index + 1}/{total_images}] Labeled HF{filename}: {description_hf}")
            else:
                print(f"[{index + 1}/{total_images}] Failed to label {filename}")

    print(f"\nSummary: {labeled_count} new images labeled, {already_labeled_count} images already labeled.")

def save_embedding(conn, md5, gemini_embedding, huggingface_embedding):
    """Saves the embeddings for a given image (identified by md5) to the database."""
    cursor = conn.cursor()

    try:
        # Convert the embeddings to bytes for storage
        gemini_embedding_bytes = gemini_embedding.tobytes() if isinstance(gemini_embedding, np.ndarray) else None
        huggingface_embedding_bytes = huggingface_embedding.tobytes() if isinstance(huggingface_embedding, np.ndarray) else None

        if gemini_embedding_bytes is None or huggingface_embedding_bytes is None:
            raise ValueError("One or both embeddings are invalid and cannot be saved.")

        cursor.execute("""
            INSERT OR REPLACE INTO embeddings (md5, gemini_embedding, huggingface_embedding)
            VALUES (?, ?, ?)
        """, (md5, gemini_embedding_bytes, huggingface_embedding_bytes))
        conn.commit()
    except Exception as e:
        print(f"Error saving embedding for {md5}: {e}")
        conn.rollback()

def get_embedding(conn, md5):
    """Retrieves the embeddings for a given image (identified by md5) from the database."""
    cursor = conn.cursor()
    cursor.execute("""
        SELECT gemini_embedding, huggingface_embedding FROM embeddings WHERE md5 = ?
    """, (md5,))
    result = cursor.fetchone()
    if result:
        # Convert the bytes back to NumPy arrays
        gemini_embedding = np.frombuffer(result[0], dtype=np.float32) if result[0] else None
        huggingface_embedding = np.frombuffer(result[1], dtype=np.float32) if result[1] else None
        return gemini_embedding, huggingface_embedding
    return None, None

def retrieve_captions(conn):
    """Retrieves the images in the SQL DB and checks if they already exist in Milvus, returns the non-existent ones for embedding."""
    cursor = conn.cursor()

    query = f"SELECT * FROM captions WHERE md5 NOT IN (SELECT md5 FROM embeddings)"
    cursor.execute(query)

    infos = cursor.fetchall()
    print(infos)
    if not infos:
        raise Exception("No images")
    else:
        print(infos)
        return infos
    
def clean_embeddings_test(conn):
    """
    Deletes all entries in the embeddings table where either gemini_embedding or huggingface_embedding is invalid.
    """
    cursor = conn.cursor()

    try:
        # Retrieve all embeddings
        cursor.execute("SELECT md5, gemini_embedding, huggingface_embedding FROM embeddings")
        rows = cursor.fetchall()

        invalid_md5s = []

        for row in rows:
            md5, gemini_blob, huggingface_blob = row

            # Validate BLOB sizes
            if gemini_blob and len(gemini_blob) % 4 != 0:
                print(f"Invalid Gemini embedding size for MD5: {md5}")
                invalid_md5s.append(md5)
                continue
            if huggingface_blob and len(huggingface_blob) % 4 != 0:
                print(f"Invalid Hugging Face embedding size for MD5: {md5}")
                invalid_md5s.append(md5)
                continue

            # Convert BLOBs back to NumPy arrays
            gemini_embedding = np.frombuffer(gemini_blob, dtype=np.float32) if gemini_blob else None
            huggingface_embedding = np.frombuffer(huggingface_blob, dtype=np.float32) if huggingface_blob else None

            # Check if either embedding is invalid (None or empty)
            if gemini_embedding is None or huggingface_embedding is None or len(gemini_embedding) == 0 or len(huggingface_embedding) == 0:
                print(f"Invalid embedding found for MD5: {md5}")
                invalid_md5s.append(md5)

        # Delete invalid embeddings
        if invalid_md5s:
            placeholders = ','.join('?' for _ in invalid_md5s)
            query = f"DELETE FROM embeddings WHERE md5 IN ({placeholders})"
            cursor.execute(query, invalid_md5s)
            conn.commit()
            print(f"Deleted {len(invalid_md5s)} invalid embeddings.")
        else:
            print("No invalid embeddings found.")

    except sqlite3.Error as e:
        print(f"Error cleaning embeddings: {e}")
        conn.rollback()

    
def retrieve_embeddings(conn):
    """
    Retrieves the embeddings and deserializes them back into NumPy arrays.
    """
    cursor = conn.cursor()
    query = "SELECT * FROM embeddings"
    cursor.execute(query)
    infos = cursor.fetchall()

    embeddings = []
    for res in infos:
        md5 = res[0]
        gemini_embedding = np.frombuffer(res[1], dtype=np.float32) if res[1] else None
        huggingface_embedding = np.frombuffer(res[2], dtype=np.float32) if res[2] else None

        if gemini_embedding is None or huggingface_embedding is None:
            print(f"Invalid embeddings found for MD5 {md5}. Skipping...")
            continue

        embeddings.append((gemini_embedding, huggingface_embedding))

    if not embeddings:
        raise Exception("No valid embeddings found in the database.")
    else:
        print(f"Retrieved {len(embeddings)} embeddings from the database.")
        return embeddings

def drop_database():
    """Deletes the SQLite database file."""
    if os.path.isfile("labels.db"):
        os.remove("labels.db")
    print("SQLite database deleted.")


def retrieve_images(conn, hashes):
    """Retrieves the images in the SQL DB and checks if they already exist in Milvus, returns the non-existent ones for embedding."""
    cursor = conn.cursor()

    placeholders = ','.join('?' for _ in hashes)
    hashes = list(hashes)
    query = f"SELECT * FROM images WHERE md5 NOT IN ({placeholders})"
    cursor.execute(query, hashes)

    infos = cursor.fetchall()
    if not infos:
        raise Exception("No images to label")
    else:
        return infos

def retrieve_all_images(conn):
    """Retrieves the images in the SQL DB"""
    cursor = conn.cursor()

    query = f"SELECT * FROM images"
    cursor.execute(query)
    
    infos = cursor.fetchall()
    if not infos:
        raise Exception("No images to label")
    else:
        return infos


class ImageInformation:
    """Class to store and retrieve image information."""

    def __init__(self, md5, path, description):
        self.md5 = md5
        self.path = path
        self.description = description

    def get_info(self):
        return self.md5, self.path, self.description


def get_all_labels(conn, prompt):
    """Fetches all image paths and their labels from the DB for a specific prompt."""
    cursor = conn.cursor()
    cursor.execute("SELECT image_path, label FROM images WHERE prompt = ?", (prompt,))
    return dict(cursor.fetchall())
