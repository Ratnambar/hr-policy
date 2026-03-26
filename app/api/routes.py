from fastapi import FastAPI
from app.core.rag_engine import rag_chain, llm
from app.rag.ingestor import split_text
from app.rag.vectorstore import vectorstore
from app.rag.embeddings import create_embeddings
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision
from datasets import Dataset
import math

app = FastAPI()

def sanitize_score(score_dict):
    clean = {}
    for k, v in score_dict.items():
        try:
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                clean[k] = None
            else:
                clean[k] = v
        except Exception:
            clean[k] = None
    return clean



@app.get("/ingest")
def ingest_route():
    chunks = split_text()
    vs = vectorstore()
    vs.add_documents(chunks)
    return {"chunks": chunks}

@app.post("/ask_question")
def ask_question_route(question: str):
    results = []
    result = rag_chain.invoke({"query": question})
    print("\n Answer: ")
    print("\n Source Chunks Used:")
    answer = result["result"]
    # print("\n Answer: ", answer)
    if not answer:
        return {
            "score": None,
            "error": "LLM returned an empty answer - cannot evaluate."
        }
    contexts = [doc.page_content for doc in result["source_documents"]]
    contexts = [c for c in contexts if c.strip()]
    if not contexts:
        return {
            "score": None,
            "error": "No source chunks retrieved — cannot evaluate context metrics."
        }
    manual_ground_truth = (
        "Leaves without approval are Loss of Pay (LOP). "
        "Unused EL can be encashed up to 30 days in Dec or at resignation."
    )
    results.append({
            "question": question,
            "answer": answer.strip(),
            "contexts": contexts,
            "ground_truth": manual_ground_truth
        })
    dataset = Dataset.from_list(results)
    score = evaluate(
        dataset,
        llm=llm,
        embeddings=create_embeddings(),
        metrics=[faithfulness, answer_relevancy, context_recall, context_precision]
    )
    score_dict = score.to_pandas().to_dict(orient="records")[0]
    clean_score = sanitize_score(score_dict)
    return {
        "answer": answer.replace("\n\n", "").strip(),
        "score": {"faithfulness": clean_score["faithfulness"], "answer_relevancy": clean_score["answer_relevancy"], "context_recall": clean_score["context_recall"], "context_precision": clean_score["context_precision"]}
    }
        # "sources": [
        #     {
        #         "page": doc.metadata.get("page", "?"),
        #         "content": doc.page_content[:200]
        #     }
        #     for doc in result["source_documents"]
        # ]