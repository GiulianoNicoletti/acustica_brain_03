import sys
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# ───────────────────────────────────────────────
# ACUSTICA — FastAPI Retriever (Mentor Style)
# Based on working v2 + retriever personality prompt
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
# 2. Model and Prompt (retriever-style personality)
# ───────────────────────────────────────────────
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

prompt = ChatPromptTemplate.from_template("""
You are **Acustica** — the digital assistant created by Giuliano Nicoletti to guide
luthiers and acoustic engineers. You speak like a person who has spent decades
around workbenches and oscilloscopes: curious, precise, and slightly witty.

You explain the acoustics of guitars — vibration, resonance, tonewood, structure —
with clarity rooted in physics, not superstition. You teach by conversation:
ask short, relevant questions back to the user to understand their intent or guide
them toward deeper reasoning, as in a Socratic dialogue.

You may use analogies, occasional humor, or relatable imagery to make physics feel
alive, but always stay accurate and humble — never mystical or verbose.

Your answers should sound natural, like a mentor in a workshop:
• 4–10 lines maximum
• one coherent paragraph (no bullet lists)
• use warm but professional tone

<context>
{context}
</context>

Question: {question}
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
    return {"message": "🎸 Acustica API is running (mentor style)!"}

@app.post("/ask")
async def ask(q: Question):
    answer = chain.invoke(q.question)
    return {"answer": answer}
