from dotenv import load_dotenv
import os
from google import genai

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

result = client.models.embed_content(
    model="gemini-embedding-001",
    contents="Binary search is used on sortd arrays"
)

embedding = result.embeddings[0].values
print(f"Embedding ki length (kitne numbers hain): {len(embedding)}")
print(f"Pehle 10 numbers: {embedding[:10]}")