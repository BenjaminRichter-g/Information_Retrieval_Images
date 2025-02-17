import gemini_api as ga
import sys
import os
import sqlite3

def init_db(db_path="labels.db"):
    """Initializes the SQLite database and creates the images table if it doesn't exist."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
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
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            full_path = os.path.join(directory, filename)
            # Check if this image is already labeled
            cursor.execute("SELECT id FROM images WHERE image_path = ?", (full_path,))
            if cursor.fetchone() is None:
                # Call the model to label the image
                description = model.imageQuery(full_path, "Describe what is in this image in one sentence.")
                if description:
                    cursor.execute(
                        "INSERT INTO images (image_path, label) VALUES (?, ?)",
                        (full_path, description)
                    )
                    conn.commit()
                    print(f"Labeled {filename}: {description}")
                else:
                    print(f"Failed to label {filename}")
            else:
                print(f"Already labeled {filename}")

def main():
    model = ga.ModelApi()
    args = sys.argv[1:]

    # Check for the "-create-label" flag
    create_label = any(flag in args for flag in ["-create-label"])

    if create_label:
        try:
            directory = "images/"
        except (ValueError, IndexError):
            print("Please provide a directory path after the -dir flag.")
            return

        conn = init_db()
        label_images(directory, model, conn)
        conn.close()
    else:
        print("No operation specified.")

if __name__ == "__main__":
    main()

