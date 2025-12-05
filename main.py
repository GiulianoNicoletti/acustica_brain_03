# ───────────────────────────────────────────────
# ACUSTICA — FastAPI Retriever (Debug Version)
# This version logs vectorstore path & contents
# ───────────────────────────────────────────────

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from pathlib import Path
import os

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

# Debug logs (to check Render deployment)
print("VECTORSTORE PATH:", VECTOR_DIR)
if VECTOR_DIR.exists():
    print("VECTORSTORE CONTENTS:", os.listdir(VECTOR_DIR))
else:
    print("VECTORSTORE DIRECTORY MISSING!")

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise EnvironmentError("Missing OPENAI_API_KEY in .env file")

# Load embeddings and Chroma vectorstore
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma(
    collection_name="acustica_corpus_v1",
    embedding_function=embeddings,
    persist_directory=str(VECTOR_DIR)
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# Define model
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

# Prompt template
prompt = ChatPromptTemplate.from_template("""
You are Acustica, assistant for luthiers and acoustic engineers.
Use the retrieved context to answer clearly and precisely.

Context:
{context}

Question:
{question}
""")

# Retrieval + LLM chain
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

# Allow CORS for testing (temporary)
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

# Home route
@app.get("/")
def home():
    return {"message": "🎸 Acustica API is running!"}

# Ask route
@app.post("/ask")
async def ask(q: Question):
    answer = chain.invoke(q.question)
    return {"answer": answer}
