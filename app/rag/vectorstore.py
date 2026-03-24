import os
from langchain_pinecone import PineconeVectorStore
from .embeddings import create_embeddings, create_index
from dotenv import load_dotenv
load_dotenv()

def vectorstore():
    vectorstore = PineconeVectorStore(
        index_name=create_index(),
        embedding=create_embeddings(),
        pinecone_api_key=os.environ.get("PINECONE_API_KEY")
    )
    return vectorstore