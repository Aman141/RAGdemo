import json
from typing import List, Dict, Optional

import pypdf
from langchain_text_splitters import RecursiveCharacterTextSplitter


def pdf_loader(file_path: str) -> Optional[pypdf.PdfReader]:
	"""Load a PDF and return a PdfReader, or None on failure."""
	try:
		return pypdf.PdfReader(file_path)
	except Exception as exc:  # noqa: BLE001
		print(f"Error reading PDF: {exc}")
		return None


def split_text(text: str, chunk_size: int = 600, chunk_overlap: int = 100) -> List[str]:
	"""Split text into semantically meaningful chunks."""
	splitter = RecursiveCharacterTextSplitter(
		chunk_size=chunk_size,
		chunk_overlap=chunk_overlap,
		separators=["\n\n", "\n", ".", "!", "?", " ", ""],
	)
	return splitter.split_text(text)


def create_chunks(pdf_reader: pypdf.PdfReader, chunk_size: int = 600) -> List[Dict]:
	"""Create chunks from each page of a PDF."""
	chunks: List[Dict] = []
	for page_index, page in enumerate(pdf_reader.pages):
		text = page.extract_text()
		if not text:
			print(f"Warning: Page {page_index} has no extractable text.")
			continue
		text = text.strip()
		page_chunks = split_text(text, chunk_size=chunk_size)
		for chunk_index, chunk in enumerate(page_chunks):
			chunks.append(
				{
					"file_name": getattr(pdf_reader, "metadata", {}).get("/Title", "unknown"),
					"page_index": page_index,
					"chunk_index": chunk_index,
					"text": chunk,
				}
			)
	return chunks


def save_chunks(chunks: List[Dict], output_path: str) -> None:
	"""Save chunks to a JSON file."""
	with open(output_path, "w") as file_handle:
		json.dump(chunks, file_handle)
	print(f"Chunks saved to {output_path}")


def run_ingestion(pdf_path: str, output_path: str) -> None:
	"""Run the ingestion process from PDF to chunks JSON."""
	reader = pdf_loader(pdf_path)
	if reader:
		chunks = create_chunks(reader, chunk_size=600)
		save_chunks(chunks, output_path)


