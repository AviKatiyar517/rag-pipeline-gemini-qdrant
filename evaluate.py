from dotenv import load_dotenv
import os
from google import genai
import numpy as np
import faiss
import re
from eval_testset import test_cases

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)


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


print("Setting up retrieval system...")
all_chunks = load_and_chunk_documents("docs")

embeddings = []
for chunk in all_chunks:
    emb = get_embedding(chunk["text"])
    embeddings.append(emb)

embeddings_array = np.array(embeddings)
dimension = embeddings_array.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings_array)

print(f"Setup done. {len(all_chunks)} chunks indexed.\n")

print("=== Running Evaluation ===\n")
correct = 0
total = len(test_cases)

for test in test_cases:
    question = test["question"]
    expected = test["expected_source"]

    query_embedding = get_embedding(question).reshape(1, -1)
    distances, indices = index.search(query_embedding, 1)  # sirf top-1 chahiye

    retrieved_filename = all_chunks[indices[0][0]]["filename"]

    is_correct = (retrieved_filename == expected)
    status = "CORRECT" if is_correct else "WRONG"

    if is_correct:
        correct += 1

    print(f"[{status}] Q: {question}")
    print(f"   Expected: {expected} | Got: {retrieved_filename}\n")

accuracy = (correct / total) * 100
print(f"=== Final Accuracy: {correct}/{total} = {accuracy:.1f}% ===")