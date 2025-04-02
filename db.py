import sqlite3
import os
from hashlib import md5

def init_db(db_path="labels.db"):
    """Initializes the SQLite database and creates the images table if it doesn't exist."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS images (
            md5 TEXT PRIMARY KEY,
            image_path TEXT UNIQUE,
            label TEXT
        )
    ''')
    conn.commit()
    return conn

def label_images(directory, model, conn):
    """Iterates over image files in the given directory, labels those not already in the database, and stores the results."""
    cursor = conn.cursor()
    for filename in os.listdir(directory):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.heic')):
            full_path = os.path.join(directory, filename)
            # Check if this image is already labeled
            cursor.execute("SELECT md5 FROM images WHERE image_path = ?", (full_path,))
            if cursor.fetchone() is None:
                # Call the model to label the image
                description = model.imageQuery(full_path, "Describe what is in this image in one sentence.")
                if description:
                    with open(full_path, 'rb') as f:
                        file_data = f.read()
                        file_hash = md5(file_data).hexdigest()

                    cursor.execute(
                        "INSERT INTO images (md5, image_path, label) VALUES (?, ?, ?)",
                        (file_hash, full_path, description)
                    )
                    conn.commit()
                    print(f"Labeled {filename}: {description}")
                else:
                    print(f"Failed to label {filename}")
            else:
                print(f"Already labeled {filename}")



def retrieve_images(conn, hashes):
    """Retrieves the images in the SQL DB and checks if they already exist in Milvus, returns the non-existent ones for embedding."""
    cursor = conn.cursor()
    
    placeholders = ','.join('?' for _ in hashes)
    query = f"SELECT * FROM images WHERE md5 NOT IN ({placeholders})"
    cursor.execute(query, hashes)
    
    infos = cursor.fetchall()
    if not infos:
        raise Exception("No images to label")
    else:
        print(infos)
        return infos


class ImageInformation():

    def __init__(self, md5, path, description):
        self.md5 = md5
        self.path = path
        self.description = description


    def get_info(self):

        return self.md5, self.path, self.description


