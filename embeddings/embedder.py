from langchain_community.embeddings import OpenAIEmbeddings
from typing import List, Dict
from dotenv import load_dotenv
import os

load_dotenv()

def load_embedding_model(model_name: str = "text-embedding-3-small"):
    """Load embedding model from OpenAI."""
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")
    return OpenAIEmbeddings(model=model_name, api_key=key)


def embed_chunks(chunks: List[Dict], model: OpenAIEmbeddings, batch_size: int = 2) -> List[Dict]:
    """
    Embed chunks in batches to avoid hitting API quota limits.
    If you are running out of quota, consider:
      - Reducing the number of chunks
      - Using a smaller batch size
      - Using a local embedding model (e.g., sentence-transformers)
    """
    import time

    texts = [chunk["text"] for chunk in chunks]
    embedded_chunks = []
    total = len(texts)

    for i in range(0, total, batch_size):
        batch_texts = texts[i:i+batch_size]
        batch_chunks = chunks[i:i+batch_size]
        try:
            vectors = model.embed_documents(batch_texts)
        except Exception as e:
            # If quota error, print and break
            print(f"Embedding failed at batch {i//batch_size+1}: {e}")
            print("Consider using a local embedding model if you keep hitting quota limits.")
            break
        for chunk, vector in zip(batch_chunks, vectors):
            embedded_chunk = {
                "file_name": chunk["file_name"],
                "page_index": chunk["page_index"],
                "chunk_index": chunk["chunk_index"],
                "text": chunk["text"],
                "embedding": vector
            }
            embedded_chunks.append(embedded_chunk)
        # Optional: sleep to avoid rate limits
        time.sleep(0.5)
    return embedded_chunks

# Alternative: Use a local embedding model (uncomment and install sentence-transformers)
# from sentence_transformers import SentenceTransformer
# def embed_chunks_local(chunks: List[Dict], model_name: str = "all-MiniLM-L6-v2") -> List[Dict]:
#     model = SentenceTransformer(model_name)
#     texts = [chunk["text"] for chunk in chunks]
#     vectors = model.encode(texts, show_progress_bar=True)
#     embedded_chunks = []
#     for chunk, vector in zip(chunks, vectors):
#         embedded_chunk = {
#             "file_name": chunk["file_name"],
#             "page_index": chunk["page_index"],
#             "chunk_index": chunk["chunk_index"],
#             "text": chunk["text"],
#             "embedding": vector.tolist()
#         }
#         embedded_chunks.append(embedded_chunk)
#     return embedded_chunks

