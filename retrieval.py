from dotenv import load_dotenv
import os
from google import genai
import numpy as np

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

def load_documents(folder_path):
    documents = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                documents.append({
                    "filename": filename,
                    "content": content
                })

    return documents

def get_embedding(text):
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text
    )
    return np.array(result.embeddings[0].values)

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot_product / (norm1 * norm2)

docs = load_documents("docs")
print(f"Loaded {len(docs)} documents\n")

for doc in docs:
    doc["embedding"] = get_embedding(doc["content"])
    print(f"Embedded: {doc['filename']}")


query = input("\nApna question poocho: ")

query_embedding = get_embedding(query)


print("\n--- Similarity Scores ---")
scores = []
for doc in docs:
    similarity = cosine_similarity(query_embedding, doc["embedding"])
    scores.append((doc["filename"], similarity, doc["content"]))
    print(f"{doc['filename']}: {similarity:.4f}")


scores.sort(key=lambda x: x[1], reverse=True)
best_match = scores[0]

print(f"\n Sabse relevant document: {best_match[0]} (score: {best_match[1]:.4f})")
print(f"Content: {best_match[2]}")

context = best_match[2]

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