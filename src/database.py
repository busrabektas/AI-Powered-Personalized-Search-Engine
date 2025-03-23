import json
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams
from src.config import QDRANT_URL, COLLECTION_NAME

client = QdrantClient(QDRANT_URL)

def setup_qdrant():
    """Create Qdrant collection."""
    if COLLECTION_NAME not in client.get_collections().collections:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)  
        )
        print(f"'{COLLECTION_NAME}' collection created.")
    else:
        print(f"✅ '{COLLECTION_NAME}' collection already exists.")

def upload_embeddings(json_file="data/cleaned_wikipedia_embeddings.json"):
    """Load Embeddings to Qdrant Database."""
    with open(json_file, "r", encoding="utf-8") as f:
        articles = json.load(f)

    points = [
        PointStruct(
            id=int(article["id"]),
            vector=article["embedding"],
            payload={"title": article["title"], "url": article["url"], "text": article["text"]}
        )
        for article in articles
    ]

    client.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"{len(articles)} article successfully upoaded.")
