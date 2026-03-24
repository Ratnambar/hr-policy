import os
from langchain_huggingface import HuggingFaceEndpoint
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from app.rag.vectorstore import vectorstore
from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()

vectorstore = vectorstore()


llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.environ.get("GROQ_API_KEY"),
    temperature=0.3,
    max_tokens=512,
)

prompt = PromptTemplate(
     input_variables=["context", "question"],
    template="""You are an HR assistant. Use ONLY the context below to answer.
    If the answer is not in the context, say "I don't know based on the provided documents."

    Context:
    {context}
    Question: {question}

    Give a clear, detailed answer with bullet points where appropriate.
    Answer:""",
)

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)

rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt}

)