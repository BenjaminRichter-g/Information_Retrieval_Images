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

    def get_all_md5_hashes(self):
        expr = "md5 != ''"  # any valid filtering on your text field
        query_results = self.collection.query(expr=expr, output_fields=["md5"])
        md5_hashes = {record["md5"] for record in query_results if "md5" in record}
        return md5_hashes