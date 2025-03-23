from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
import time
from src.config import QDRANT_URL, COLLECTION_NAME

client = QdrantClient(QDRANT_URL)

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


VECTOR_SIZE = model.get_sentence_embedding_dimension()

def ensure_collection_exists():
    """Koleksiyon yoksa oluşturur."""
    existing_collections = client.get_collections()
    collection_names = [c.name for c in existing_collections.collections]
    
    if COLLECTION_NAME not in collection_names:
        print(f"Collection '{COLLECTION_NAME}' not exists. New collection is creating...")
        
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=VECTOR_SIZE, 
                distance=Distance.COSINE  
            )
        )
        
        time.sleep(2)
        print(f"Collection '{COLLECTION_NAME}' created successfully.")
    else:
        print(f"Collection '{COLLECTION_NAME}' already exists.")


def search_qdrant(query: str, top_k=5):
    """Returns Wikipedia articles most similar to the user's text query"""
    
    ensure_collection_exists()

    query_vector = model.encode(query).tolist()

    search_result = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=top_k  
    )

    if not search_result:
        return []

    return search_result  
