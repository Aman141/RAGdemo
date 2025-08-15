from typing import List, Dict
import faiss
import numpy as np
from pathlib import Path

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



