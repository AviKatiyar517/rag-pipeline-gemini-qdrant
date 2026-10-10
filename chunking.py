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

with open("docs/neural_networks.txt", "r", encoding="utf-8") as f:
    content = f.read()

chunks = chunk_text(content, max_chunk_size=300, overlap_sentences=1)

print(f"Total chunks banaye: {len(chunks)}\n")

for i, chunk in enumerate(chunks):
    print(f"--- Chunk {i+1} (length: {len(chunk)}) ---")
    print(chunk)
    print()