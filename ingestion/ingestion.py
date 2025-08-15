import pypdf

def pdf_loader(file_path):
    pdf_reader = pypdf.PdfReader(file_path)
    return pdf_reader


def create_chuncks(pdf_reader, chunk_size=600):
    chunks = []
    for page_index, page in enumerate(pdf_reader.pages):
        text = page.extract_text()

        chunk_pages = [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

        for chunk_index, chunk in enumerate(chunk_pages):
            chunks.append({
                "page_index": page_index,
                "chunk_index": chunk_index,
                "text": chunk
            })
    return chunks


if __name__ == "__main__":
    pdf_reader = pdf_loader("data/raw/meidtations.pdf")
    chunks = create_chuncks(pdf_reader)
    print(chunks)