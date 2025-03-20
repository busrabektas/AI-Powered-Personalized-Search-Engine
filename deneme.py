from datasets import load_dataset
import json

# Wikimedia Wikipedia Türkçe verisetini yükle
dataset = load_dataset("wikimedia/wikipedia", "20231101.en", split="train[:2]")

# İlk makaleyi al
first_article = dataset[0]

# JSON formatında dosya olarak kaydet
file_path = "first_article.json"
with open(file_path, "w", encoding="utf-8") as f:
    json.dump(first_article, f, ensure_ascii=False, indent=4)
