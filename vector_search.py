from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient


client = QdrantClient("http://localhost:6333")  # Qdrant local çalışıyorsa


model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

collection_name = "test_collection"
def search_qdrant(query: str, top_k=5):
    """Kullanıcının metin sorgusuna en benzer Wikipedia makalelerini getirir."""
    
    query_vector = model.encode(query).tolist()

    search_result = client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=top_k  
    )

    for i, result in enumerate(search_result):
        payload = result.payload
        print(f"{i+1}. {payload['title']} - {payload['url']} ({result.score:.4f})\n")
        print(payload['text'][:500] + "...\n")
        print("-" * 80)

query_text = "History of anarchism"
search_qdrant(query_text, top_k=3)
