from langchain_community.embeddings import OpenAIEmbeddings
from typing import List, Dict

def load_embeddings(model_name: str = "text-embedding-3-small"):
    """Load embeddings from OpenAI."""
    return OpenAIEmbeddings(model=model_name)


def embed_chunks(chunks: List[Dict], embeddings) -> List[Dict]:
    """Embed chunks."""
    texts = [chunk["text"] for chunk in chunks]

    try:
        vectors = embeddings.embed_documents(texts)
    except Exception as e:
        raise RuntimeError(f"Embedding failed: {e}")
    
    embedded_chunks = []
    for chunk, vector in zip(chunks, vectors):
        embedded_chunk = {
            "file_name": chunk["file_name"],
            "page_index": chunk["page_index"],
            "chunk_index": chunk["chunk_index"],
            "text": chunk["text"],
            "embedding": vector
        }
        embedded_chunks.append(embedded_chunk)
    return embedded_chunks

