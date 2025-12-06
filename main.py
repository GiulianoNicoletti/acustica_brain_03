import sys
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# ───────────────────────────────────────────────
# ACUSTICA — Conversational Retriever with Context Synthesis
# Author: Giuliano Nicoletti
# Purpose: coherent, physics-grounded reasoning from corpus
# Multilingual version — automatic language detection and translation
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

# 🆕 Multilingual support
from langchain_openai import ChatOpenAI as ChatTranslator

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
You are **Acustica** — the digital assistant created by Giuliano Nicoletti to
guide luthiers and acoustic engineers. You speak as a thoughtful craftsman who
has spent decades around workbenches, instruments, and oscilloscopes.

Your role is to help the user understand one concept at a time. Be clear,
conversational, and grounded in physics — never overwhelming, never speculative.
If the retrieved context does not clearly define the concept, say so honestly and
do not invent or guess beyond what the corpus provides. If the user introduces a
new topic, connect ideas only when they are explicitly related; do not jump ahead
or create associations that have not been mentioned.

Your tone is warm, professional, and precise — like an experienced teacher in a
quiet workshop. Offer insight through gentle guidance rather than lectures.

Style guidelines:
• 4–8 sentences maximum  
• one coherent paragraph  
• plain, natural language (no bullet lists)  
• end with a short, relevant follow-up question that invites reflection  

──────────────────────────────────────────────
Conversation so far:
{history}
──────────────────────────────────────────────
<context>
{context}
</context>

Question: {question}
""")

# ───────────────────────────────────────────────
# 3. Multilingual translation utilities
# ───────────────────────────────────────────────
translator_detect = ChatTranslator(model="gpt-4o-mini", temperature=0)
translator_translate = ChatTranslator(model="gpt-4o-mini", temperature=0)

def detect_language(text: str) -> str:
    """Return ISO language code (e.g., en, it, fr, es, de, ja)."""
    result = translator_detect.invoke(
        f"Detect the language of this text and reply only with its ISO code:\n{text}"
    )
    return result.content.strip().lower()

def translate_if_needed_to_english(text: str) -> str:
    """Translate any language into English for retrieval alignment."""
    lang = detect_language(text)
    if lang.startswith("en"):
        return text
    translated = translator_translate.invoke(
        f"Translate this text into clear, technical English for acoustic and lutherie contexts:\n{text}"
    )
    return translated.content.strip()

def translate_back_if_needed(answer: str, original_text: str) -> str:
    """Translate generated English answer back to the user's original language."""
    lang = detect_language(original_text)
    if lang.startswith("en"):
        return answer
    back = translator_translate.invoke(
        f"Translate this into {lang}, preserving all acoustic and physical terminology precisely:\n{answer}"
    )
    return back.content.strip()

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
# 4. FastAPI app
# ───────────────────────────────────────────────
app = FastAPI(title="Acustica — Conversational Reasoning API (Multilingual)")

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
    return {"message": "🎸 Acustica — Conversational Reasoning API (Multilingual) running!"}

@app.post("/ask")
async def ask(q: Question):
    # 🆕 Translate to English for retrieval
    translated_question = translate_if_needed_to_english(q.question)

    # Normal reasoning chain (unchanged)
    answer = chain.invoke(translated_question)
    memory.save_context({"question": q.question}, {"answer": answer})

    # 🆕 Translate back to user language if needed
    final_answer = translate_back_if_needed(answer, q.question)
    return {"answer": final_answer}
