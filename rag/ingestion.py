from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_pdf(pdf_paths: list[str]) -> list:
    all_documents = []
    for path in pdf_paths:
        try:
            loader = PyPDFLoader(path)
            docs = loader.load()
            all_documents.extend(docs)
        except Exception as e:
            print(f"Error loading {path}: {e}")
    return all_documents

def text_splitting(documents: list, chunk_size: int = 750, chunk_overlap: int = 150) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    return splitter.split_documents(documents)