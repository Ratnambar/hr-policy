import os
from dotenv import load_dotenv
load_dotenv()

langsmith_api_key = os.environ.get("LANGSMITH_API_KEY")
pinecone_api_key = os.environ.get("PINECONE_API_KEY")