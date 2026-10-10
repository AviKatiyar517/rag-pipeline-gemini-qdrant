from dotenv import load_dotenv
import os
from google import genai
import numpy as np
import re
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)
qdrant = QdrantClient(path="qdrant_data")

COLLECTION_NAME = "rag_docs"


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
            current_chunk = current_chunk[-1:] if current_chunk else []
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
                    all_chunks.append({"filename": filename, "text": chunk})
    return all_chunks

print("Loading documents...")
all_chunks = load_and_chunk_documents("docs")
print(f"Total chunks: {len(all_chunks)}")

# Step 2: Embeddings banao
print("Generating embeddings...")
embeddings = []
for chunk in all_chunks:
    emb = get_embedding(chunk["text"])
    embeddings.append(emb)

dimension = len(embeddings[0])

if not qdrant.collection_exists(COLLECTION_NAME):
    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=dimension, distance=Distance.COSINE)
    )
    print("Collection created.")
else:
    print("Collection already exists.")

points = []
for i, (chunk, emb) in enumerate(zip(all_chunks, embeddings)):
    points.append(
        PointStruct(
            id=i,
            vector=emb.tolist(),
            payload={"filename": chunk["filename"], "text": chunk["text"]}
        )
    )

qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
print(f"{len(points)} chunks stored in Qdrant.\n")

query = input("Apna question poocho: ")
query_embedding = get_embedding(query)

results = qdrant.query_points(
    collection_name=COLLECTION_NAME,
    query=query_embedding.tolist(),
    limit=3
).points

print("\n--- Top 3 results ---")
for r in results:
    print(f"\nScore: {r.score:.4f}")
    print(f"From: {r.payload['filename']}")
    print(f"Text: {r.payload['text']}")