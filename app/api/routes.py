from textwrap import indent
from fastapi import FastAPI
from app.core.rag_engine import rag_chain
from app.rag.ingestor import ingest_pdf, split_text
from app.rag.vectorstore import vectorstore

app = FastAPI()

@app.get("/ingest")
def ingest_route():
    chunks = split_text()
    vs = vectorstore()
    vs.add_documents(chunks)
    return {"chunks": chunks}

@app.post("/ask_question")
def ask_question_route(question: str):
    result = rag_chain.invoke({"query": question})
    print("\n Answer: ")
    print("\n Source Chunks Used:")
    return {
        "answer": result["result"],
        "sources": [
            {
                "page": doc.metadata.get("page", "?"),
                "content": doc.page_content[:200]
            }
            for doc in result["source_documents"]
        ]
    }