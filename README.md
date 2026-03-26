# HR Policy Assistant — RAG-powered Q&A System

> Ask any question about your company's HR policies and get instant, accurate answers with source references — built with HuggingFace, Pinecone, and LangChain.

---

## Demo

![HR Policy RAG Demo]

```
User:  "How many casual leaves am I entitled to per year?"
Bot:   "As per the Leave Policy (Section 3.2), every confirmed employee is
        entitled to 12 casual leaves per calendar year. Leaves cannot be
        carried forward to the next year.
        Source: HR_Policy_2024.pdf — Page 8"
```

---

## Problem Statement

HR teams in Indian companies receive hundreds of repetitive policy questions every month — about leaves, appraisals, reimbursements, notice periods, and more. Employees waste time waiting for HR replies. HR teams waste time answering the same questions repeatedly.

This project solves that by letting employees get instant, accurate answers directly from the official HR policy documents — with citations, so they can verify themselves.

---

## Features

- Ask questions in plain English about any HR policy
- Answers always cite the exact page and section from the source document
- Says "I don't know" when the answer is not in the document — no hallucination
- Supports multiple PDF documents (leave policy, appraisal policy, code of conduct, etc.)
- Simple web UI built with Streamlit — no technical knowledge needed to use
- Fully free stack — no OpenAI or paid API required

---

## Tech Stack

| Component | Tool | Why |
|---|---|---|
| Embedding model | `BAAI/bge-small-en-v1.5` (HuggingFace) | Free, runs locally, 384-dim |
| Vector database | Pinecone (free tier) | Scalable, managed, easy to use |
| LLM | `mistralai/Mistral-7B-Instruct-v0.3` (HuggingFace Inference API) | Free, strong instruction following |
| Orchestration | LangChain | Retrieval chains, prompt management |
| UI | Streamlit | Fast to build, easy to demo |
| Evaluation | RAGAS | Industry-standard RAG eval metrics |

---

## Architecture

```
                    INGESTION (run once)
┌─────────────┐    ┌──────────────┐    ┌──────────────┐    ┌───────────────┐
│  HR Policy  │───▶│  Chunk docs  │───▶│ Embed chunks │───▶│   Pinecone    │
│  PDF files  │    │ 512 tok/50   │    │  bge-small   │    │  Vector Index │
└─────────────┘    └──────────────┘    └──────────────┘    └───────────────┘

                    QUERY (per user request)
┌─────────────┐    ┌──────────────┐    ┌──────────────┐    ┌───────────────┐
│  User query │───▶│ Embed query  │───▶│ Retrieve top │───▶│  LLM answer   │
│             │    │  bge-small   │    │   5 chunks   │    │  + citations  │
└─────────────┘    └──────────────┘    └──────────────┘    └───────────────┘
```

---

## Evaluation Results

Evaluated on **30 HR policy questions** with manually verified ground truth answers using [RAGAS](https://github.com/explodinggradients/ragas).

| Metric | Score | What it measures |
|---|---|---|
| Faithfulness | **0.33** | Are answers grounded in the document? |
| Answer relevancy | **0.95** | Does the answer address the question? |
| Context recall | **0.79** | Did retrieval find the right chunks? |
| Context precision | **0.83** | Were retrieved chunks actually relevant? |

### What these scores mean

- **Faithfulness 0.33** — 33% of the time, the LLM's answer is fully supported by the retrieved context. It is not making things up.
- **Answer relevancy 0.95** — 95% of answers directly address what the user asked. Very few off-topic responses.
- **Context recall 0.79** — In 79% of cases, the retrieval step found all the chunks needed to answer correctly. Improving chunk size or re-ranking could push this higher.
- **Context precision 0.83** — 83% of retrieved chunks were genuinely useful for answering the question. Low noise in retrieval.

### Honest limitations

- Scores were measured on a single company's HR document. Results may vary on other documents.
- The HuggingFace free inference API occasionally times out under load — for production use, run the model locally.
- Context recall drops on questions that span multiple sections of the document (e.g. "What are all the types of leave?").

> Run the evaluation yourself: `python main.py` actually evaluation is calculating in `ask_question` route in `routes.py`

---


---

## Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/yourusername/hr-policy-rag.git
cd hr-policy-rag
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up environment variables

```bash
cp .env.example .env
# Edit .env and add your keys
```

```env
PINECONE_API_KEY=your_pinecone_key_here
HF_TOKEN=your_huggingface_token_here
```

Get your free keys:
- Pinecone: [pinecone.io](https://pinecone.io) → free tier
- HuggingFace: [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

### 4. Ingest your HR policy PDF

```bash
- python ingestor.py --ingest data/hr_policy.pdf
- give your pdf file path "file_path = r"your pdf file path"
- hit "/ingest" route for index creating, embeddings, chunking
```

### 5. Run the app

```bash
python main.py
```

Open [http://localhost:8000] in your browser.

---

## Example Questions You Can Ask

```
"What happens if an employee takes leave without approval?",
"What percentage of basic salary does the employer contribute to PF?",
"How long is the probation period for new hires at NovaTech?",
"What increment percentage does a Rating 5 employee receive?",
"How many days does an employee have to respond to a Show Cause Notice?",
"What is the gift value limit before an employee needs approval under the conflict of interest policy?"
```

---

## Run Evaluation

```bash
python main.py
- evaluation code is under "/ask_question" route so when someone ask any question, evaluation will be process.
```

## Requirements

```
- run "pip install -r requirements.txt" this will install dependecies
```

---

## What I Learned Building This

- Chunk size and overlap have a bigger impact on quality than the choice of LLM
- Writing a proper eval dataset before tuning anything is essential — without it you're guessing
- The `faithfulness` metric is the most important one for an HR use case — employees need to trust the answers
- RAGAS needs an LLM to compute metrics — it uses the HuggingFace inference API by default

---
