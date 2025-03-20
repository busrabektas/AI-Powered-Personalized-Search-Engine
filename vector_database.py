import json
import unicodedata
import re
from datasets import load_dataset
from sentence_transformers import SentenceTransformer


# 📌 Modeli yükle
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Örnek veri yükleme (İlk 5 makaleyi almak için)
dataset = load_dataset("wikimedia/wikipedia", "20231101.en", split="train[:5]")

def clean_unicode(text):
    """Unicode karakterlerini normalleştir ve gereksizleri temizle."""
    text = unicodedata.normalize("NFKC", text)  # Unicode normalizasyonu
    text = re.sub(r"\s+", " ", text)  # Gereksiz boşlukları temizle
    text = re.sub(r"[^\w\s.,!?()\"'-]", "", text)  # Özel sembolleri temizle
    return text.strip()

# 📌 Dataset'i temizleyip embedding ekleyerek liste formatına çevir
cleaned_articles = []
for article in dataset:
    cleaned_text = clean_unicode(article["text"])  # Unicode temizliği uygulanmış metin
    embedding = model.encode(cleaned_text).tolist()  # Metni embedding'e çevir

    cleaned_articles.append({
        "id": article["id"],
        "title": article["title"],
        "url": article["url"],
        "text": cleaned_text,
        "embedding": embedding  # Embedding ekledik
    })

# JSON dosyasına kaydet
file_path = "cleaned_wikipedia_embeddings.json"
with open(file_path, "w", encoding="utf-8") as f:
    json.dump(cleaned_articles, f, ensure_ascii=False, indent=4)

print(f"Unicode karakterleri düzeltildi, embedding'ler eklendi ve '{file_path}' dosyasına kaydedildi.")
