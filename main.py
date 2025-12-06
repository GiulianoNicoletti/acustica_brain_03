import sys
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# ───────────────────────────────────────────────
# ACUSTICA — Multilingual Conversational Retriever (Stable Core)
# Base: validated mentor-style version by Giuliano Nicoletti
# Upgrade: automatic language detection + translation in/out
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
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

print("🧠 Checking Chroma collections…")
try:
    collections = vectorstore._client.list_collections()
    print("Available collections:", [c.name for c in collections])
except Exception as e:
    print("Error listing collections:", e)

# ───────────────────────────────────────────────
# 2. LLM, Memory, Prompt (unchanged mentor tone)
# ───────────────────────────────────────────────
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
memory = ConversationBufferMemory(memory_key="history", input_key="question", return_messages=False)

prompt = ChatPromptTemplate.from_template("""
You are **Acustica** — the digital assistant created by Giuliano Nicoletti to guide
luthiers and acoustic engineers. You speak as a thoughtful craftsman who has spent
decades around workbenches, instruments, and oscilloscopes.

Your role is to help the user understand one concept at a time. Be clear,
conversational, and grounded in physics — never overwhelming, never speculative.
If the retrieved context does not clearly define the concept, say so honestly and
do not invent or guess beyond what the corpus provides.

Tone: warm, professional, precise — like an experienced teacher in a quiet workshop.
Offer insight through gentle guidance rather than lectures.

Style:
• 4–8 sentences maximum  
• single coherent paragraph  
• natural technical language (no bullet lists)  
• end with a short, relevant question inviting reflection

──────────────────────────────────────────────
Conversation so far:
{history}
──────────────────────────────────────────────
<context>
{context}
</context>

Question: {question}
""")

chain = (
    {"context": retriever, "question": RunnablePassthrough(), "history": lambda _: memory.load_memory_variables({}).get("history", "")}
    | prompt
    | llm
    | StrOutputParser()
)

# ───────────────────────────────────────────────
# 3. Multilingual utilities
# ───────────────────────────────────────────────
translator = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def detect_language(text: str) -> str:
    result = translator.invoke(f"Detect the language of this text and respond only with its ISO code:\n{text}")
    return result.content.strip().lower()

def translate_to_english_if_needed(text: str) -> tuple[str, str]:
    lang = detect_language(text)
    if lang.startswith("en"):
        return text, "en"
    translated = translator.invoke(f"Translate this text into clear, technical English:\n{text}")
    return translated.content.strip(), lang

def translate_back(answer: str, lang: str) -> str:
    if lang.startswith("en"):
        return answer
    back = translator.invoke(f"Translate this text into {lang}, preserving all acoustic terminology:\n{answer}")
    return back.content.strip()

# ───────────────────────────────────────────────
# 4. FastAPI App
# ───────────────────────────────────────────────
app = FastAPI(title="Acustica — Multilingual Conversational Retriever")

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
    return {"message": "🎸 Acustica — Multilingual Conversational Retriever running!"}

@app.post("/ask")
async def ask(q: Question):
    translated_q, lang = translate_to_english_if_needed(q.question)
    answer = chain.invoke(translated_q)
    memory.save_context({"question": q.question}, {"answer": answer})
    final = translate_back(answer, lang)
    return {"answer": final}
