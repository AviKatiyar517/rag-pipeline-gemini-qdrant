from dotenv import load_dotenv
import os
from google import genai
import numpy as np
import faiss
import re

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)


import re

def split_into_sentences(text):
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s for s in sentences if s]


def chunk_text(text, max_chunk_size=300, overlap_sentences=1):
    sentences = split_into_sentences(text)
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        sentence_length = len(sentence)
        if current_length + sentence_length > max_chunk_size and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = current_chunk[-overlap_sentences:] if overlap_sentences > 0 else []
            current_length = sum(len(s) for s in current_chunk)
        current_chunk.append(sentence)
        current_length += sentence_length

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def get_embedding(text):
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text
    )
    return np.array(result.embeddings[0].values, dtype="float32")


def load_and_chunk_documents(folder_path):
    all_chunks = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                chunks = chunk_text(content)
                for chunk in chunks:
                    all_chunks.append({
                        "filename": filename,
                        "text": chunk
                    })
    return all_chunks

print("Documents load aur chunk ho rahe hain...")
all_chunks = load_and_chunk_documents("docs")
print(f"Total chunks: {len(all_chunks)}\n")

print("Embeddings generate ho rahi hain...")
embeddings = []
for chunk in all_chunks:
    emb = get_embedding(chunk["text"])
    embeddings.append(emb)

embeddings_array = np.array(embeddings)
print(f"Embeddings shape: {embeddings_array.shape}\n")


dimension = embeddings_array.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings_array)
print(f"FAISS index mein {index.ntotal} chunks add ho gaye\n")

query = input("Apna question poocho: ")
query_embedding = get_embedding(query).reshape(1, -1)

k=3
distances, indices = index.search(query_embedding, k)

print(f"\n--- Top {k} relevant chunks ---")
for rank, idx in enumerate(indices[0]):
    chunk_info = all_chunks[idx]
    print(f"\nRank {rank+1} (distance: {distances[0][rank]:.4f})")
    print(f"From: {chunk_info['filename']}")
    print(f"Text: {chunk_info['text']}")

best_distance = distances[0][0]
RELEVANCE_THRESHOLD = 0.85  

if best_distance > RELEVANCE_THRESHOLD:
    print("\n=== FINAL ANSWER (RAG) ===")
    print("I don't have enough information in my documents to answer this question.")
else:
    context_parts = []
    for idx in indices[0]:
        context_parts.append(all_chunks[idx]["text"])

    context = "\n\n".join(context_parts)

    augmented_prompt = f"""Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't have enough information to answer this."

Context:
{context}

Question: {query}

Answer:"""

    final_response = client.models.generate_content(
        model="gemini-3.8-flash",
        contents=augmented_prompt
    )

    print("\n=== FINAL ANSWER (RAG) ===")
    print(final_response.text)

augmented_prompt = f"""Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't have enough information to answer this."

Context:
{context}

Question: {query}

Answer:"""

final_response = client.models.generate_content(
    model="gemini-3.8-flash",
    contents=augmented_prompt
)

print("\n=== FINAL ANSWER (RAG) ===")
print(final_response.text)