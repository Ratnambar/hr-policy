import os
from pinecone import Pinecone
from langchain_pinecone import PineconeEmbeddings
from dotenv import load_dotenv
load_dotenv()


api_key=os.environ.get("PINECONE_API_KEY")
pc = Pinecone(api_key=api_key)
index_name = "hr-policy"

# Pinecone integrated inference only allows these models for create_index_for_model:
# 'llama-text-embed-v2', 'multilingual-e5-large', 'pinecone-sparse-english-v0'
# (HuggingFace IDs like BAAI/bge-small-en-v1.5 are NOT supported here.)
PINECONE_EMBED_MODEL = "multilingual-e5-large"


def create_index():
    """Create the Pinecone index if missing. field_map must be string field names, not chunks."""
    existing = [idx.name for idx in pc.list_indexes()]
    if index_name not in existing:
        pc.create_index_for_model(
            name=index_name,
            cloud="aws",
            region="us-east-1",
            embed={
                "model": PINECONE_EMBED_MODEL,
                # Maps embed input "text" -> record field that holds the chunk text (use same key when upserting)
                "field_map": {"text": "text"},
            },
        )
    return index_name

def create_embeddings():
    embeddings = PineconeEmbeddings(
        model=PINECONE_EMBED_MODEL,
        pinecone_api_key=os.environ.get("PINECONE_API_KEY")
    )
    return embeddings