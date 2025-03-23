from src.search import search_qdrant
from openai import OpenAI
import os


openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)


def generate_answer(query: str, top_k=3, max_context_chars=4000):

    results = search_qdrant(query, top_k=top_k)

    if not results:
        return "No relevant documents found."

    context_parts = []
    sources = []
    total_chars = 0

    for res in results:
        title = res.payload.get("title", "Unknown Title")
        text = res.payload.get("text", "")
        url = res.payload.get("url", "")

        source_block = f"Title: {title}\nURL: {url}\nText: {text}"
        if total_chars + len(source_block) > max_context_chars:
            remaining = max_context_chars - total_chars
            if remaining > 0:
                context_parts.append(source_block[:remaining])
                sources.append((title, url))
            break

        context_parts.append(source_block)
        sources.append((title, url))
        total_chars += len(source_block)

    context = "\n\n".join(context_parts)

    prompt = f"""
You are an expert assistant. Based on the following Wikipedia sources, answer the user's question.

Context:
{context}

Question: {query}
Answer:"""

    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )

    answer = completion.choices[0].message.content.strip()

    return answer, sources
