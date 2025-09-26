import pypdf
from langchain_text_splitters import RecursiveCharacterTextSplitter
import json

def pdf_loader(file_path):
    """Load PDF using PyPDF and return the reader object."""
    try:
        pdf_reader = pypdf.PdfReader(file_path)
        return pdf_reader
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return None


def split_text(text, chunk_size=600, chunk_overlap=100):
    """Split text into semantically meaningful chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", "!", "?", " ", ""]
    )
    return splitter.split_text(text)


def create_chunks(pdf_reader, chunk_size=600):
    """Split each page's text into chunks of specified character size."""
    chunks = []

    for page_index, page in enumerate(pdf_reader.pages):
        text = page.extract_text()

        if not text:
            print(f"Warning: Page {page_index} has no extractable text.")
            continue

        text = text.strip()
        page_chunks = split_text(text, chunk_size=chunk_size)

        for chunk_index, chunk in enumerate(page_chunks):
                chunks.append({
                    "file_name": pdf_reader.metadata["/Title"],
                    "page_index": page_index,
                    "chunk_index": chunk_index,
                    "text": chunk
                })

    return chunks

def save_chunks(chunks, output_path):
    """Save chunks to a JSON file."""
    with open(output_path, "w") as f:
        json.dump(chunks, f)
    print(f"Chunks saved to {output_path}")


def run_ingestion(pdf_path, output_path):
    """Run the ingestion process."""
    reader = pdf_loader(pdf_path)
    if reader:
        chunks = create_chunks(reader, chunk_size=600)
        save_chunks(chunks, output_path)



if __name__ == "__main__":
    pdf_path = "data/raw/meidtations.pdf"
    output_path = "data/processed/meidtations.json"
    run_ingestion(pdf_path, output_path)
