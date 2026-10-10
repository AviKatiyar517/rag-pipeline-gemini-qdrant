from fastapi import FastAPI
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

app = FastAPI()

qdrant = QdrantClient(path="qdrant_data")
COLLECTION_NAME = "rag_docs"
RELEVANCE_THRESHOLD = 0.5  
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

print("Setting up RAG system with Qdrant...")

if not qdrant.collection_exists(COLLECTION_NAME):
    print("Collection nahi mili, naya data load/embed kar rahe hain...")
    all_chunks = load_and_chunk_documents("docs")
    embeddings = []
    for chunk in all_chunks:
        emb = get_embedding(chunk["text"])
        embeddings.append(emb)

    dimension = len(embeddings[0])
    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=dimension, distance=Distance.COSINE)
    )

    points = []
    for i, (chunk, emb) in enumerate(zip(all_chunks, embeddings)):
        points.append(
            PointStruct(id=i, vector=emb.tolist(), payload={"filename": chunk["filename"], "text": chunk["text"]})
        )
    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"{len(points)} chunks stored.")
else:
    print("Existing collection mil gayi, dobara embed nahi karna pad raha!")

print("Server ready!\n")


@app.get("/query")
def query_rag(question: str):
    try:
        query_embedding = get_embedding(question)

        results = qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=query_embedding.tolist(),
            limit=3
        ).points

        best_score = results[0].score if results else 0

        if best_score < RELEVANCE_THRESHOLD:
            return {
                "question": question,
                "answer": "I don't have enough information in my documents to answer this question.",
                "sources": []
            }

        context_parts = []
        sources = []
        for r in results:
            context_parts.append(r.payload["text"])
            sources.append(r.payload["filename"])

        context = "\n\n".join(context_parts)

        augmented_prompt = f"""Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't have enough information to answer this."

Context:
{context}

Question: {question}

Answer:"""

        response = client.models.generate_content(
            model="gemini-3.8-flash",
            contents=augmented_prompt
        )

        return {
            "question": question,
            "answer": response.text,
            "sources": list(set(sources))
        }

    except Exception as e:
        return {
            "question": question,
            "answer": "Sorry, something went wrong. Please try again.",
            "error": str(e),
            "sources": []
        }