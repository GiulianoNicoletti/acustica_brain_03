# ───────────────────────────────────────────────
# Acustica Brain — Phase 3 Retriever (Conversational Socratic version)
# Giuliano Nicoletti — December 2025
# ───────────────────────────────────────────────

from pathlib import Path
from dotenv import load_dotenv
import os
import sys

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# ───────────────────────────────────────────────
# 1. Setup paths, environment, and flags
# ───────────────────────────────────────────────
DEBUG_MODE = "--debug" in sys.argv

BASE_DIR = Path(__file__).resolve().parent
VECTOR_DIR = BASE_DIR / "vectorstore"
COLLECTION_NAME = "acustica_corpus_v2"   # Updated to new version

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    key_file = BASE_DIR / "Acustica_API_Key.txt"
    if key_file.exists():
        with open(key_file, "r", encoding="utf-8") as f:
            os.environ["OPENAI_API_KEY"] = f.read().strip()
    else:
        raise EnvironmentError("❌ No API key found (.env or Acustica_API_Key.txt)")

# ───────────────────────────────────────────────
# 2. Load vectorstore and model
# ───────────────────────────────────────────────
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
db = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=str(VECTOR_DIR)
)

retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": 10, "lambda_mult": 0.5})
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.35, max_tokens=400)

# ───────────────────────────────────────────────
# 3. Socratic, human prompt definition
# ───────────────────────────────────────────────
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

# ───────────────────────────────────────────────
# 4. Chain setup
# ───────────────────────────────────────────────
chain = (
    {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# ───────────────────────────────────────────────
# 5. Interactive CLI loop with optional debug
# ───────────────────────────────────────────────
print("\n🎸  Acustica Brain — Phase 3 Retriever (Conversational Socratic)")
if DEBUG_MODE:
    print("🔍 Debug mode ON — full retrieved context will be shown\n")
else:
    print("Run with '--debug' to display full retrieved text for each query.\n")

print("Ask about acoustics, design, or luthiery (type 'exit' to quit)\n")

while True:
    query = input("❓ Ask › ").strip()
    if query.lower() in ["exit", "quit", "q"]:
        print("👋 Bye! Keep building and listening.")
        break

    print("\n💭 Thinking...\n")

    results = db.similarity_search_with_score(query, k=10)
    docs = [r[0] for r in results]
    scores = [r[1] for r in results]

    context_text = "\n---\n".join([d.page_content for d in docs])
    answer = chain.invoke({"context": context_text, "question": query})
    print(f"💬 {answer}\n")

    print("— Top source chunks —")
    for i, (doc, score) in enumerate(results, start=1):
        src = doc.metadata.get("source", "unknown")
        snippet = doc.page_content[:180].replace("\n", " ")
        conf = round((1 - score) * 100, 1)
        print(f"{i}. [{src}]  ({conf}% match)  {snippet}...")
    print("------------------------------------------------------------")

    if DEBUG_MODE:
        print("\n📚 Full retrieved context:")
        print("────────────────────────────")
        for i, doc in enumerate(docs, start=1):
            src = doc.metadata.get("source", "unknown")
            print(f"[{i}] {src}\n{doc.page_content}\n")
        print("------------------------------------------------------------\n")
