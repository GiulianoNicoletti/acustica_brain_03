import sys
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# ───────────────────────────────────────────────
# ACUSTICA — Conversational Retriever with Context Synthesis
# Author: Giuliano Nicoletti
# Purpose: coherent, physics-grounded reasoning from corpus
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
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain.memory import ConversationBufferMemory

# ───────────────────────────────────────────────
# 1. Setup
# ───────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
VECTOR_DIR = BASE_DIR / "vectorstore"

print("VECTORSTORE PATH:", VECTOR_DIR)
if VECTOR_DIR.exists():
    print("VECTORSTORE CONTENTS:", os.listdir(VECTOR_DIR))
else:
    print("VECTORSTORE DIRECTORY MISSING!")

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise EnvironmentError("Missing OPENAI_API_KEY in .env file")

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma(
    collection_name="acustica_corpus_v2",
    embedding_function=embeddings,
    persist_directory=str(VECTOR_DIR)
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

print("🧠 Checking Chroma collections…")
try:
    collections = vectorstore._client.list_collections()
    print("Available collections:", [c.name for c in collections])
except Exception as e:
    print("Error listing collections:", e)

# ───────────────────────────────────────────────
# 2. LLM, Memory, Context Synthesizer, Prompt
# ───────────────────────────────────────────────
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

memory = ConversationBufferMemory(
    memory_key="history",
    input_key="question",
    return_messages=False
)

# ─────────────── Context synthesis layer ───────────────
def synthesize_context(docs):
    """Fuse retrieved chunks into one coherent technical summary."""
    joined = "\n\n".join(d.page_content for d in docs)
    if not joined.strip():
        return ""
    summarizer = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    synthesis_prompt = f"""
    Combine and integrate the following excerpts into one coherent technical summary.
    Focus on the physics and acoustic principles without repetition or speculation.
    Keep only factual, explanatory content — no lists, no fluff.
    ---
    {joined}
    """
    response = summarizer.invoke(synthesis_prompt)
    return response.content.strip()

# ─────────────── Conversational mentor prompt ───────────────
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
• warm, professional tone

──────────────────────────────────────────────
Conversation so far:
{history}
──────────────────────────────────────────────
<context>
{context}
</context>

Question: {question}
""")

# ─────────────── Retrieval + synthesis + LLM chain ───────────────
chain = (
    {
        "context": retriever | RunnableLambda(synthesize_context),
        "question": RunnablePassthrough(),
        "history": lambda _: memory.load_memory_variables({}).get("history", "")
    }
    | prompt
    | llm
    | StrOutputParser()
)

# ───────────────────────────────────────────────
# 3. FastAPI app
# ───────────────────────────────────────────────
app = FastAPI(title="Acustica — Conversational Reasoning API")

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
    return {"message": "🎸 Acustica — Conversational Reasoning API running!"}

@app.post("/ask")
async def ask(q: Question):
    answer = chain.invoke(q.question)
    memory.save_context({"question": q.question}, {"answer": answer})
    return {"answer": answer}
