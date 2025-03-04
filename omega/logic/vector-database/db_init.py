from pymilvus import MilvusClient, DataType
import os
import backend.retrieval.vector_db as vdb

__LOCAL_DB: str = os.path.join(
    os.path.dirname(__file__), "..", "backend", "retrieval", "milvus.db"
)

client = MilvusClient(uri="http://localhost:19530")

schema = MilvusClient.create_schema(
    auto_id=False,
    enable_dynamic_field=True,
)

schema.add_field(
    field_name="id", datatype=DataType.VARCHAR, max_length=36, is_primary=True
)
schema.add_field(field_name="chunk", datatype=DataType.FLOAT_VECTOR, dim=768)
schema.add_field(field_name="chunk_raw", datatype=DataType.VARCHAR, max_length=1000)
schema.add_field(field_name="lecture", datatype=DataType.VARCHAR, max_length=10000)
schema.add_field(field_name="lecture_no", datatype=DataType.INT8)

index_params = client.prepare_index_params()

index_params.add_index(field_name="chunk", index_type="AUTOINDEX", metric_type="COSINE")

client.drop_collection(collection_name="lectures")
client.create_collection(
    collection_name="lectures", schema=schema, index_params=index_params
)

res = client.get_load_state(collection_name="lectures")

print(res)
