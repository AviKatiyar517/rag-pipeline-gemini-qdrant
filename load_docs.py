import os

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
docs = load_documents("docs")

print(f"Total documents loaded: {len(docs)}\n")

for doc in docs:
    print(f"--- {doc['filename']} ---")
    print(doc['content'][:100] + "...")
    print()