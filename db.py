import sqlite3
import os
from hashlib import md5

def init_db(db_path="labels.db"):
    """Initializes the SQLite database and creates the images table if it doesn't exist."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_path TEXT UNIQUE,
            label TEXT,
            md5 TEXT
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
            cursor.execute("SELECT id FROM images WHERE image_path = ?", (full_path,))
            if cursor.fetchone() is None:
                # Call the model to label the image
                description = model.imageQuery(full_path, "Describe what is in this image in one sentence.")
                if description:
                    with open(full_path, 'rb') as f:
                        file_data = f.read()
                        file_hash = md5(file_data).hexdigest()

                    cursor.execute(
                        "INSERT INTO images (image_path, label, md5) VALUES (?, ?, ?)",
                        (full_path, description, file_hash)
                    )
                    conn.commit()
                    print(f"Labeled {filename}: {description}")
                else:
                    print(f"Failed to label {filename}")
            else:
                print(f"Already labeled {filename}")


