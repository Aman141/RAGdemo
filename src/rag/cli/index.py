import argparse
import json
from pathlib import Path

from rag.embeddings.embedder import load_embedding_model, embed_chunks
from rag.embeddings.vector_store import build_index


def main() -> None:
	parser = argparse.ArgumentParser(description="Build FAISS index from chunks JSON")
	parser.add_argument("chunks", type=Path, help="Path to chunks JSON (from ingestion)")
	parser.add_argument("index", type=Path, help="Output FAISS index path")
	parser.add_argument("--model", default=None, help="Embedding model name (optional)")
	parser.add_argument("--batch-size", type=int, default=2, help="Batch size for embedding")
	args = parser.parse_args()

	with open(args.chunks, "r") as file_handle:
		chunks = json.load(file_handle)

	model = load_embedding_model(model_name=args.model or "text-embedding-3-small")
	embedded = embed_chunks(chunks=chunks, model=model, batch_size=args.batch_size)
	build_index(embedded_chunks=embedded, index_path=args.index)


if __name__ == "__main__":
	main()


