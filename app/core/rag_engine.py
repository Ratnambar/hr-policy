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
    template="""You are an HR Assistant for NovaTech Solutions. Your job is to answer employee questions clearly, 
        warmly, and accurately using only the provided HR policy documents.

        TONE & STYLE:
        - Be friendly and professional — like a helpful HR colleague, not a search engine
        - Use conversational prose, not raw bullet dumps
        - Bold key numbers and policy names for scannability
        - Always cite the policy name/number you're referencing

        STRUCTURE (follow this order):
        1. Brief warm acknowledgment of the question (1 sentence)
        2. Direct answer in prose — lead with the most relevant info first
        3. If the answer varies by employee type, ask "Are you permanent or contract staff?" 
        before giving numbers — or present both clearly labeled
        4. Source reference: "This is per [Policy Name, HRP-XXX], Page X"
        5. Follow-up invitation: offer 1-2 related things they might want to know next

        ACCURACY RULES:
        - Only answer from the retrieved policy context provided below
        - If information isn't in the context, say: "I don't have that in the current policy. 
        Please contact HR at hr@novatech.com"
        - Never guess or hallucinate policy details

        CONTEXT FROM POLICY DOCS:
        {context}

        USER QUESTION:
        {question}"""


    # template="""You are an HR assistant. Use ONLY the context below to answer.
    # If the answer is not in the context, say "I don't know based on the provided documents."

    # Context:
    # {context}
    # Question: {question}

    # Give a clear, detailed answer with bullet points where appropriate.
    # Answer:""",
)

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 1}
)

rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt}

)