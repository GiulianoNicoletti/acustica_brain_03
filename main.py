# ───────────────────────────────────────────────
# ACUSTICA — FastAPI Retriever (Stable Cloud Version)
# Retrieves from local ChromaDB (vectorstore) and answers via OpenAI
# ───────────────────────────────────────────────

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from pathlib import Path
import os

# LangChain imports
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# ───────────────────────────────────────────────
# 1. Setup
# ───────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
VECTOR_DIR = BASE_DIR / "vectorstore"

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise EnvironmentError("Missing OPENAI_API_KEY in .env file")

# Disable Chroma telemetry (Render blocks analytics, this fixes CollectionQueryEvent error)
import chromadb
chromadb.config.settings.anonymized_telemetry = False

# Load vectorstore
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma(
    collection_name="acustica_corpus_v1",
    embedding_function=embeddings,
    persist_directory=str(VECTOR_DIR)
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# LLM setup
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

# Prompt template
prompt = ChatPromptTemplate.from_template("""
You are Acustica, assistant for luthiers and acoustic engineers.
Use the retrieved context to answer clearly, precisely, and technically.
If the context is missing, respond that no indexed documents were found.

Context:
{context}

Question:
{question}
""")

# Chain definition
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# ───────────────────────────────────────────────
# 2. FastAPI App
# ───────────────────────────────────────────────
app = FastAPI(title="Acustica API")

# Allow requests from any domain (for testing)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Question(BaseModel):
    question: str

@app.get("/")
def home():
    return {"message": "🎸 Acustica API is running and connected to vectorstore!"}

@app.post("/ask")
async def ask(q: Question):
    answer = chain.invoke(q.question)
    return {"answer": answer}

# ───────────────────────────────────────────────
# Run command for local testing (not needed on Render)
# uvicorn main:app --host 0.0.0.0 --port 8000
# ───────────────────────────────────────────────
