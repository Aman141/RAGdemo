import os
from typing import List, Dict

from dotenv import load_dotenv
from langchain_community.embeddings import OpenAIEmbeddings


load_dotenv()


def load_embedding_model(model_name: str = "text-embedding-3-small") -> OpenAIEmbeddings:
	"""Load an OpenAI embedding model configured via env vars."""
	key = os.getenv("OPENAI_API_KEY")
	if not key:
		raise ValueError("OPENAI_API_KEY not found in environment variables")
	return OpenAIEmbeddings(model=model_name, api_key=key)


def embed_chunks(chunks: List[Dict], model: OpenAIEmbeddings, batch_size: int = 2) -> List[Dict]:
	"""Embed chunk texts in small batches to respect rate limits."""
	import time

	texts = [chunk["text"] for chunk in chunks]
	embedded_chunks: List[Dict] = []
	total = len(texts)

	for start_index in range(0, total, batch_size):
		batch_texts = texts[start_index : start_index + batch_size]
		batch_chunks = chunks[start_index : start_index + batch_size]
		try:
			vectors = model.embed_documents(batch_texts)
		except Exception as exc:  # noqa: BLE001
			print(f"Embedding failed at batch {start_index // batch_size + 1}: {exc}")
			print("Consider using a local embedding model if you keep hitting quota limits.")
			break
		for chunk, vector in zip(batch_chunks, vectors):
			embedded_chunk = {
				"file_name": chunk.get("file_name"),
				"page_index": chunk.get("page_index"),
				"chunk_index": chunk.get("chunk_index"),
				"text": chunk.get("text"),
				"embedding": vector,
			}
			embedded_chunks.append(embedded_chunk)
		time.sleep(0.5)

	return embedded_chunks


