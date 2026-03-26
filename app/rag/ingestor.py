from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def ingest_pdf():
    file_path = r"C:\Users\Ratnambar\Downloads\HR_Policy_Document_NovaTech.pdf"
    if not file_path:
        raise ValueError("File path is required")
    else:
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        return docs

def split_text():
    docs = ingest_pdf()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        add_start_index=True
    )
    chunks = text_splitter.split_documents(docs)
    return chunks