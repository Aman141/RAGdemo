from pathlib import Path
from typing import List, Dict

import faiss
import numpy as np


def build_index(embedded_chunks: List[Dict], index_path: Path) -> None:
	"""Build and persist a FAISS index (inner product similarity)."""
	if not embedded_chunks:
		raise ValueError("No embedded chunks provided.")

	embeddings = np.array([chunk["embedding"] for chunk in embedded_chunks], dtype="float32")
	dimension = embeddings.shape[1]
	index = faiss.IndexFlatIP(dimension)
	index.add(embeddings)
	faiss.write_index(index, str(index_path))
	print(f"✅ FAISS index built and saved to {index_path}")


