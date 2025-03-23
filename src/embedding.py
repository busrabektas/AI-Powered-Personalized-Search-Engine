import json
import unicodedata
import re
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

dataset = load_dataset("wikimedia/wikipedia", "20231101.en", split="train[:5]")

def clean_text(text):
    """Unicode karakterlerini normalleştir ve gereksiz sembolleri temizle."""
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text)  
    text = re.sub(r"[^\w\s.,!?()\"'-]", "", text) 
    return text.strip()

def generate_embeddings(output_file="data/cleaned_wikipedia_embeddings.json"):
    """Wikipedia verisini temizleyip embedding'leri oluşturur."""
    cleaned_articles = []
    for article in dataset:
        cleaned_text = clean_text(article["text"])
        embedding = model.encode(cleaned_text).tolist()  

        cleaned_articles.append({
            "id": article["id"],
            "title": article["title"],
            "url": article["url"],
            "text": cleaned_text,
            "embedding": embedding
        })

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(cleaned_articles, f, ensure_ascii=False, indent=4)

    print(f"Embeddings are created and saved to '{output_file}' .")
