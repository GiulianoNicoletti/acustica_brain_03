import sys
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# ───────────────────────────────────────────────
# ACUSTICA — FastAPI Conversational Retriever
# Based on working v2 (with collection debug)
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
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain

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
# 2. Model, Memory, and Prompt
# ───────────────────────────────────────────────
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)

memory = ConversationBufferMemory(
    memory_key="history",
    input_key="question",
    return_messages=False
)

prompt = ChatPromptTemplate.from_template("""
You are Acustica, assistant for luthiers and acoustic engineers,
created by Giuliano Nicoletti. Use retrieved technical context
when available; if not, answer from your deep knowledge of
guitar acoustics and tonewood physics.

Be detailed, structured, and natural — not curt. Write in full
sentences suited for technical readers. At the end, propose ONE
short, relevant follow-up question to keep the discussion alive.

──────────────────────────────────────────────
Conversation so far:
{history}
──────────────────────────────────────────────
Retrieved context:
{context}
──────────────────────────────────────────────
User:
{question}
──────────────────────────────────────────────
Answer:
""")

# Define retrieval + LLM chain with memory
chain = (
    {
        "context": retriever | (lambda docs: "\n\n".join(d.page_content for d in docs)),
        "question": RunnablePassthrough(),
        "history": lambda _: memory.load_memory_variables({}).get("history", "")
    }
    | prompt
    | llm
    | StrOutputParser()
)

# ───────────────────────────────────────────────
# 3. FastAPI App
# ───────────────────────────────────────────────
app = FastAPI(title="Acustica Conversational API")

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
    return {"message": "🎸 Acustica Conversational API is running!"}

@app.post("/ask")
async def ask(q: Question):
    answer = chain.invoke(q.question)
    memory.save_context({"question": q.question}, {"answer": answer})
    return {"answer": answer}
