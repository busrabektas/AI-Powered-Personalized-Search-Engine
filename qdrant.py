import json
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams

client = QdrantClient("http://localhost:6333") 

collection_name = "test_collection"

if collection_name not in client.get_collections().collections:
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE) 
    )

file_path = "cleaned_wikipedia_embeddings.json"
with open(file_path, "r", encoding="utf-8") as f:
    articles = json.load(f)

points = []
for article in articles:
    points.append(
        PointStruct(
            id=int(article["id"]),  
            vector=article["embedding"],  
            payload={ 
                "title": article["title"],
                "url": article["url"],
                "text": article["text"],
            }
        )
    )

client.upsert(
    collection_name=collection_name,
    points=points
)

print(f"{len(articles)} makale başarıyla Qdrant'a eklendi.")
