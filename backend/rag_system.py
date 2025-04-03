from langchain_community.embeddings import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http import models

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
qdrant_client = QdrantClient(path="./qdrant_data")
COLLECTION_NAME = "docs"

def setup_vector_db(documents):
    qdrant_client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE)
    )
    points = [
        models.PointStruct(id=idx, vector=embeddings.embed_query(doc), payload={"text": doc})
        for idx, doc in enumerate(documents)
    ]
    qdrant_client.upsert(collection_name=COLLECTION_NAME, points=points)
    return qdrant_client

def query_rag(query, client):
    query_vector = embeddings.embed_query(query)
    search_result = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=3
    )
    return [hit.payload["text"] for hit in search_result]