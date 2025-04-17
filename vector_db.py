from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection
)
import numpy as np

class MilvusDb:
    def __init__(self, collection_name="image_embeddings", dim=3072):
        connections.connect(uri="http://localhost:19530", token="root:Milvus")
        self.collection_name = collection_name
        self.dim = dim

        if utility.has_collection(collection_name):
            self.collection = Collection(collection_name)
            print(f"Connected to existing Milvus collection '{collection_name}'.")
        else:
            fields = [
                FieldSchema(name="md5", dtype=DataType.VARCHAR, max_length=32, is_primary=True, auto_id=False),
                FieldSchema(name="file_path", dtype=DataType.VARCHAR, max_length=256),
                FieldSchema(name="description", dtype=DataType.VARCHAR, max_length=512),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=3072)
            ]
            schema = CollectionSchema(fields, description="Image embeddings with metadata")
            self.collection = Collection(collection_name, schema)
            print(f"Created new Milvus collection '{collection_name}'.")

        self.create_index()

    def create_index(self):
        self.collection.create_index(
            field_name="embedding",
            index_params={
                "metric_type": "L2",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 1024}
            }
        )
        self.collection.load()

    def insert_record(self, md5, file_path, description, embedding):
        if not isinstance(embedding, (list, np.ndarray)) or len(embedding) != self.dim:
            print(f"Invalid embedding for {file_path}. Skipping insertion.")
            return
        if isinstance(embedding, np.ndarray):
            embedding = embedding.tolist()  # Convert NumPy array to list
        data = [[md5], [file_path], [description], [embedding]]
        result = self.collection.insert(data)
        return result

    def delete_record(self, md5):
        expr = f"md5 == '{md5}'"
        self.collection.delete(expr)
        self.collection.flush()
        print(f"Deleted record with md5: {md5}")


    def update_description(self, md5: str, new_description: str):
        """
        Update the 'description' field of the record with the given md5.
        """
        # 1. Query existing record for its other fields
        expr = f"md5 == '{md5}'"
        query_results = self.collection.query(
            expr=expr,
            output_fields=["file_path", "embedding"]
        )
        if not query_results:
            print(f"No record found with md5 '{md5}'.")
            return None

        record = query_results[0]
        file_path = record["file_path"]
        embedding = record["embedding"]

        data = [
            {
                "md5": md5,
                "file_path": file_path,
                "description": new_description,
                "embedding": embedding
            }
        ]

        res = self.collection.upsert(data)
        self.collection.flush()
        print(f"Updated description for md5 '{md5}'.")
        return res

    def search_by_embedding(self, query_embedding, limit=10):
        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
        results = self.collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=limit,
            expr=None,
            output_fields=["md5", "file_path", "description"]
        )
        return results

    def get_by_md5(self, md5: str):
        """
        Retrieve a single record matching the given md5.
        """
        expr = f"md5 == '{md5}'"
        results = self.collection.query(
            expr=expr,
            output_fields=["md5", "file_path", "description", "embedding"]
        )
        if not results:
            print(f"No record found for md5 '{md5}'.")
            return None
        return results[0]

    def get_all_md5_hashes(self):
        expr = "md5 != ''"  # any valid filtering on your text field
        query_results = self.collection.query(expr=expr, output_fields=["md5"])
        md5_hashes = list({record["md5"] for record in query_results if "md5" in record})
        return md5_hashes
