# RAG-based DSA/ML Q&A System

A Retrieval-Augmented Generation (RAG) pipeline that answers questions about Data Structures, Algorithms, and Machine Learning concepts — grounded in a custom document knowledge base, with hallucination guardrails and persistent vector storage.

## What this project does

Instead of relying purely on an LLM's training data (which can be outdated or lead to hallucinated answers), this system:

1. Retrieves the most relevant chunks from a document knowledge base using semantic search
2. Augments the user's question with that retrieved context
3. Generates a grounded answer using an LLM, strictly based on the retrieved context
4. Refuses to answer (instead of guessing) when no relevant context is found

## Architecture

```
User Question
     |
     v
Convert to embedding (Gemini embedding model)
     |
     v
Semantic search in Qdrant vector database
     |
     v
Relevance check (threshold-based guardrail)
     |
     v
Retrieve top-K relevant chunks
     |
     v
Augment prompt with retrieved context
     |
     v
Generate answer (Gemini LLM)
     |
     v
Return answer + source documents
```

## Tech stack

| Component | Tool | Why |
|---|---|---|
| LLM | Google Gemini API | Answer generation |
| Embeddings | Gemini embedding model | Converts text to vectors for semantic search |
| Vector database | Qdrant (local/embedded mode) | Persistent storage of document embeddings |
| Backend | FastAPI | Serves the RAG pipeline as a web API |
| Chunking | Custom sentence-aware chunker | Splits documents without breaking sentences mid-way |

## Key design decisions

- **Sentence-aware chunking**: Documents are split on sentence boundaries (not raw character counts) with configurable overlap, so no chunk ends on a broken word or incomplete sentence.
- **Relevance threshold guardrail**: If the best-matching chunk's similarity score falls below a threshold, the system responds with "I don't have enough information" instead of forcing the LLM to generate an answer from irrelevant context. This reduces hallucination risk and avoids unnecessary API calls.
- **Persistent vector storage**: Switched from an in-memory FAISS index to Qdrant, so embeddings are computed once and reused across server restarts instead of being regenerated every time.
- **Error handling**: API calls are wrapped in try/except blocks so a transient LLM provider failure returns a graceful error message instead of crashing the server.

## Evaluation

A test set of 10 question-answer pairs was used to measure retrieval accuracy (does the system retrieve the chunk from the correct source document). Includes both direct and indirect/paraphrased questions to test whether retrieval relies on semantic meaning rather than keyword overlap.

**Retrieval accuracy: 10/10 (100%)** on the test set.

## API usage

Once running, the API exposes a single endpoint:

```
GET /query?question=<your question>
```

Example response:

```json
{
  "question": "how do CNNs work",
  "answer": "Based on the provided context, CNNs work through the following processes: Convolutional layers apply filters to detect features such as edges, textures, and shapes. Pooling layers reduce spatial dimensions, making the network more efficient and less prone to overfitting. Backpropagation calculates gradients to update weights and reduce loss.",
  "sources": ["neural_networks.txt"]
}
```

Interactive API documentation (Swagger UI) is available at `/docs` once the server is running.

## Project structure

```
.
├── app.py                 # FastAPI application (main entry point)
├── qdrant_setup.py         # Script to initialize and populate the Qdrant collection
├── chunking.py              # Sentence-aware chunking logic (standalone/testable)
├── evaluate.py              # Retrieval evaluation script
├── eval_testset.py          # Test questions with expected source documents
├── docs/                    # Knowledge base documents (.txt files)
├── requirements.txt          # Python dependencies
└── .env                       # API keys (not committed — see setup below)
```

## Setup

1. Clone the repository
   ```
   git clone <repo-url>
   cd rag-dsa-ml-qa
   ```

2. Create a virtual environment and activate it
   ```
   python -m venv venv
   venv\Scripts\activate      # Windows
   source venv/bin/activate   # Mac/Linux
   ```

3. Install dependencies
   ```
   pip install -r requirements.txt
   ```

4. Add your Gemini API key in a `.env` file
   ```
   GEMINI_API_KEY=your_key_here
   ```

5. Populate the vector database (first-time setup)
   ```
   python qdrant_setup.py
   ```

6. Run the API server
   ```
   uvicorn app:app --reload
   ```

7. Visit `http://127.0.0.1:8000/docs` to try it out

## Future improvements

- Hybrid search (combining semantic + keyword/BM25 search)
- Automated evaluation of answer faithfulness and relevance (RAGAS)
- Response caching for repeated queries
- Dockerized deployment
- Simple frontend instead of raw JSON responses

## Live demo

[Add your deployed Render URL here once live]
