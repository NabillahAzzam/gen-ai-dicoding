from pymilvus import connections, MilvusClient, FieldSchema, CollectionSchema, DataType, Collection

connection = connections.connect(
  alias="default",
  host='localhost', # or '0.0.0.0' or 'localhost'
  port='19530'
)

client = MilvusClient(
    uri="http://localhost:19530"
)

client.create_collection(
    collection_name="rag_llm_2",
    dimension=1536
)

res = client.get_load_state(
    collection_name="rag_llm_2"
)

