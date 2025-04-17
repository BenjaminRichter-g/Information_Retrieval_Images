import sqlite3

def print_table_contents(db_path, table_name):
    """Prints the contents of a specified table in the SQLite database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        # Query the table
        cursor.execute(f"SELECT * FROM {table_name}")
        rows = cursor.fetchall()

        # Print the table contents
        print(f"\nContents of table '{table_name}':")
        for row in rows:
            print(row)
    except sqlite3.Error as e:
        print(f"Error querying table '{table_name}': {e}")
    finally:
        conn.close()

def main():
    db_path = "labels_raghav.db"  # Path to your database file

    # List of tables to query
    tables = ["images", "captions", "embeddings", "tests"]

    for table in tables:
        print_table_contents(db_path, table)

if __name__ == "__main__":
    main()