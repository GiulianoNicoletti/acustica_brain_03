import sys
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# ───────────────────────────────────────────────
# ACUSTICA — FastAPI Retriever (Strictly Grounded)
# Based on your verified working version
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

# Debug logs to confirm Render paths
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
    collection_name="acustica_corpus_v2",
    embedding_function=embeddings,
    persist_directory=str(VECTOR_DIR)
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# 🧠 Diagnostic check — see if Chroma actually loaded the collection
print("🧠 Checking Chroma collections…")
try:
    collections = vectorstore._client.list_collections()
    print("Available collections:", [c.name for c in collections])
except Exception as e:
    print("Error listing collections:", e)

# ───────────────────────────────────────────────
# 2. Model and Prompt (strict grounding)
# ───────────────────────────────────────────────
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

prompt = ChatPromptTemplate.from_template("""
You are Acustica, the technical assistant for luthiers and acoustic engineers,
created by Giuliano Nicoletti.

Use *only* the retrieved context below to answer. Do not invent, generalize, or
introduce information not contained in the corpus. If the context lacks the
necessary details, say briefly that no specific data was found in the Acustica
corpus.

Keep answers concise, factual, and faithful to the retrieved source material.
After answering, suggest one short follow-up question that stays strictly within
the topic of the retrieved content.

──────────────────────────────────────────────
Retrieved context:
{context}
──────────────────────────────────────────────
User question:
{question}
──────────────────────────────────────────────
Answer:
""")

# Retrieval + LLM chain
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# ───────────────────────────────────────────────
# 3. FastAPI App
# ───────────────────────────────────────────────
app = FastAPI(title="Acustica API")

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
    return {"message": "🎸 Acustica API is running (strictly grounded mode)!"}

@app.post("/ask")
async def ask(q: Question):
    answer = chain.invoke(q.question)
    return {"answer": answer}
