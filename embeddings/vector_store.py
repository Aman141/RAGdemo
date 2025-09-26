from typing import List, Dict
import faiss
import numpy as np
from pathlib import Path
import json
from embedder import load_embedding_model, embed_chunks


def build_index(embedded_chunks: List[Dict], index_path: Path):
    """
    Build and save a FAISS index using the inner product (dot product) similarity.

    Args:
        embedded_chunks: List of dicts with "embedding" key (vectors).
        index_path: Path to save the FAISS index file.
    """
    if not embedded_chunks:
        raise ValueError("No embedded chunks provided.")

    # Convert list of lists to NumPy array
    embeddings = np.array([chunk["embedding"] for chunk in embedded_chunks]).astype("float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # Using Inner Product (for cosine sim, normalize first)

    index.add(embeddings)
    faiss.write_index(index, str(index_path))
    print(f"✅ FAISS index built and saved to {index_path}")



if __name__=="__main__":
    # load the processed chucks
    print("Loading processed chunks...")
    processed_chucks_path = Path('data/processed/meidtations.json')
    with open(processed_chucks_path, 'r') as f:
        processed_chucks = json.load(f)
    print(f"✅ Processed chunks loaded from {processed_chucks_path}")

    # load embedding model
    model = load_embedding_model(model_name="text-embedding-3-small")
    print("✅ Embedding model loaded")

    # create embeddings
    embedded_chucks = embed_chunks(chunks=processed_chucks, model=model)

    # build index
    index_path = 'data/vector_store/meidtations.faiss'
    build_index(embedded_chunks=embedded_chucks, index_path=index_path)