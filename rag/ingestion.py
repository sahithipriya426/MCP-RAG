from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from rag.embeddings import get_embeddings


DATA_DIR = Path("data")
CHROMA_DIR = "chroma_db"


def load_pdf(pdf_paths: list[str]) -> list:
    all_documents = []

    for path in pdf_paths:
        try:
            loader = PyPDFLoader(path)
            docs = loader.load()
            all_documents.extend(docs)

            print(f"Loaded: {Path(path).name} ({len(docs)} pages)")

        except Exception as e:
            print(f"Error loading {path}: {e}")

    return all_documents


def text_splitting(
    documents: list,
    chunk_size: int = 1500,
    chunk_overlap: int = 250
) -> list:

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )

    return splitter.split_documents(documents)


def main():

    pdf_paths = [
        str(path)
        for path in DATA_DIR.glob("*.pdf")
    ]

    if not pdf_paths:
        print("No PDF files found in data/")
        return

    print(f"Found {len(pdf_paths)} PDFs")

    # Load PDFs
    documents = load_pdf(pdf_paths)

    print(f"Total pages loaded: {len(documents)}")

    # Split documents
    chunks = text_splitting(documents)

    print(f"Created {len(chunks)} chunks")

    # Create embeddings
    embeddings = get_embeddings()

    # Create ChromaDB
    print("Creating ChromaDB...")

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR
    )

    print("ChromaDB created successfully.")


if __name__ == "__main__":
    main()