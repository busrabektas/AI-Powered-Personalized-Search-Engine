import json
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams

# 📌 Qdrant Client başlat (local host için)
client = QdrantClient("http://localhost:6333")  # Qdrant local çalışıyorsa

# 📌 Koleksiyon adı
collection_name = "test_collection"

# 📌 Koleksiyonun olup olmadığını kontrol et, yoksa oluştur
if collection_name not in client.get_collections().collections:
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE)  # MiniLM-L6-v2 için vektör boyutu 384
    )

# 📌 JSON dosyasını yükle
file_path = "cleaned_wikipedia_embeddings.json"
with open(file_path, "r", encoding="utf-8") as f:
    articles = json.load(f)

# 📌 Qdrant'a eklemek için verileri düzenle
points = []
for article in articles:
    points.append(
        PointStruct(
            id=int(article["id"]),  # ID integer olmalı
            vector=article["embedding"],  # Embedding ekleniyor
            payload={  # Metadata ekliyoruz
                "title": article["title"],
                "url": article["url"],
                "text": article["text"],
            }
        )
    )

# 📌 Verileri Qdrant'a ekle
client.upsert(
    collection_name=collection_name,
    points=points
)

print(f"✅ {len(articles)} makale başarıyla Qdrant'a eklendi.")
