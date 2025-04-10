
import sqlite3
import os
from hashlib import md5
import time

def init_db(db_path="labels.db"):
    """Initializes the SQLite database and creates the images table if it doesn't exist."""
       
    db_path = os.path.abspath(db_path)  # Make path absolute
    print(f"🔍 Initializing DB at: {db_path}")  # 👈 Add this line

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
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

def label_images(directory, model, conn,prompt):
    """Iterates over image files in the given directory, labels those not already in the database, and stores the results."""
    cursor = conn.cursor()
    for filename in os.listdir(directory):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.heic')):
            full_path = os.path.join(directory, filename)
            # Check if this image is already labeled with the given prompt
            cursor.execute("SELECT md5 FROM images WHERE image_path = ? AND prompt = ?", (full_path, prompt))

            if cursor.fetchone() is None:
                # Call the model to label the image
                description = model.imageQuery(full_path, prompt)
                time.sleep(4)  # 4s delay to stay within ~15 requests/min
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

def drop_database():

    if os.path.isfile("labels.db"):
        os.remove("labels.db")
    print("Sqlite db deleted")

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
        print(infos)
        return infos

class ImageInformation():

    def __init__(self, md5, path, description):
        self.md5 = md5
        self.path = path
        self.description = description


    def get_info(self):

        return self.md5, self.path, self.description

'''
def get_all_labels(conn):
    """Fetches all image paths and their labels from the DB."""
    cursor = conn.cursor()
    cursor.execute("SELECT image_path, label FROM images")
    return dict(cursor.fetchall())  # returns {image_path: label}
'''
#modified to get label py prompt
def get_all_labels(conn, prompt):
    cursor = conn.cursor()
    cursor.execute("SELECT image_path, label FROM images WHERE prompt = ?", (prompt,))
    return dict(cursor.fetchall())