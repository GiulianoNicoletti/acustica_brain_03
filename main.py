# ───────────────────────────────────────────────
# ACUSTICA — FastAPI Retriever (Rev03 Render-Ready)
# ───────────────────────────────────────────────

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
from dotenv import load_dotenv
import os, shutil
from asyncio import to_thread

# LangChain + Chroma
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# ───────────────────────────────────────────────
# 1. Setup and configuration
# ───────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent
SRC_VECTOR_DIR = BASE_DIR / "vectorstore"

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise EnvironmentError("Missing OPENAI_API_KEY in environment")

# Select writable directory for Chroma (Render /tmp or mounted /data)
RUNTIME_VECTOR_DIR = Path(os.getenv("VECTOR_DIR", "/tmp/vectorstore"))

# Copy vectorstore from repo to runtime directory if needed
if not RUNTIME_VECTOR_DIR.exists():
    if SRC_VECTOR_DIR.exists():
        RUNTIME_VECTOR_DIR.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(SRC_VECTOR_DIR, RUNTIME_VECTOR_DIR)
    else:
        RUNTIME_VECTOR_DIR.mkdir(parents=True, exist_ok=True)

# Disable Chroma telemetry for cleaner logs
try:
    from chromadb.config import Settings as ChromaSettings
    CHROMA_SETTINGS = ChromaSettings(anonymized_telemetry=False)
except Exception:
    CHROMA_SETTINGS = None

# Initialize embeddings + retriever
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

chroma_kwargs = dict(
    collection_name="acustica_corpus_v1",
    embedding_function=embeddings,
    persist_directory=str(RUNTIME_VECTOR_DIR),
)
if CHROMA_SETTINGS:
    chroma_kwargs["client_settings"] = CHROMA_SETTINGS

vectorstore = Chroma(**chroma_kwargs)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# ───────────────────────────────────────────────
# 2. LLM + Prompt chain
# ───────────────────────────────────────────────

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

prompt = ChatPromptTemplate.from_template("""
You are Acustica, assistant for luthiers and acoustic engineers.
Use the retrieved context to answer clearly and precisely.

Context:
{context}

Question:
{question}
""")

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# ───────────────────────────────────────────────
# 3. FastAPI Application
# ───────────────────────────────────────────────

app = FastAPI(title="Acustica API")

# CORS (open for now; tighten later)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request schema
class Question(BaseModel):
    question: str

# Health and root endpoints
@app.get("/")
def home():
    return {"message": "🎸 Acustica API is running!"}

@app.get("/healthz")
def health():
    return {"ok": True}

# Async /ask endpoint — runs LangChain in background thread
@app.post("/ask")
async def ask(q: Question):
    answer = await to_thread(chain.invoke, q.question)
    return {"answer": answer}
