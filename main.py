from src.database import setup_qdrant, upload_embeddings
from src.search import search_qdrant

# generate_embeddings() 

upload_embeddings()

#setup_qdrant()

query_text = "History of anarchism"
results = search_qdrant(query_text, top_k=3)

if not results:
    print("No reuslts found.")
else:
    for i, res in enumerate(results):
        payload = res.payload
        print(f"{i+1}. {payload['title']} - {payload['url']} ({res.score:.4f})\n")
        print(payload['text'][:500] + "...\n")
        print("-" * 80)
