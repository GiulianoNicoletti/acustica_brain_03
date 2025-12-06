import sys
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# ───────────────────────────────────────────────
# ACUSTICA — Conversational + Image+Comment Reasoning API
# ───────────────────────────────────────────────
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from pathlib import Path
import os, base64

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
# 2. LLM, memory, synthesis
# ───────────────────────────────────────────────
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
memory = ConversationBufferMemory(memory_key="history", input_key="question", return_messages=False)

def synthesize_context(docs):
    joined = "\n\n".join(d.page_content for d in docs)
    if not joined.strip(): return ""
    summarizer = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    synthesis_prompt = f"Integrate the following excerpts into one coherent technical summary:\n{joined}"
    return summarizer.invoke(synthesis_prompt).content.strip()

# ───────────────────────────────────────────────
# 3. Prompt for conversational reasoning
# ───────────────────────────────────────────────
prompt = ChatPromptTemplate.from_template("""
You are **Acustica**, created by Giuliano Nicoletti to guide luthiers and acoustic engineers.
Speak as a craftsman grounded in physics — clear, factual, and warm.
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
    {
        "context": retriever | RunnableLambda(synthesize_context),
        "question": RunnablePassthrough(),
        "history": lambda _: memory.load_memory_variables({}).get("history", "")
    }
    | prompt | llm | StrOutputParser()
)

# ───────────────────────────────────────────────
# 4. Multilingual translation layer
# ───────────────────────────────────────────────
translator = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def detect_language(text:str)->str:
    out=translator.invoke(f"Detect the language of this text and reply only with its ISO code:\n{text}")
    return out.content.strip().lower()

def translate_to_en(text:str)->str:
    lang=detect_language(text)
    if lang.startswith("en"): return text
    res=translator.invoke(f"Translate this into clear technical English:\n{text}")
    return res.content.strip()

def translate_back(answer:str, original:str)->str:
    lang=detect_language(original)
    if lang.startswith("en"): return answer
    back=translator.invoke(f"Translate this into {lang}, keeping acoustic terminology precise:\n{answer}")
    return back.content.strip()

# ───────────────────────────────────────────────
# 5. FastAPI app
# ───────────────────────────────────────────────
app = FastAPI(title="Acustica — Conversational + Image+Comment API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

class Question(BaseModel):
    question: str

@app.get("/")
def home():
    return {"message": "🎸 Acustica — Conversational + Image+Comment API running!"}

@app.post("/ask")
async def ask(q: Question):
    q_en = translate_to_en(q.question)
    answer = chain.invoke(q_en)
    memory.save_context({"question": q.question}, {"answer": answer})
    return {"answer": translate_back(answer, q.question)}

# ───────────────────────────────────────────────
# 6. Spectrum + Comment analysis
# ───────────────────────────────────────────────
@app.post("/ask_image")
async def ask_image(
    file: UploadFile = File(...),
    comment: str = Form(None)
):
    """Analyze uploaded spectrum images, optionally using user comment."""
    try:
        content = await file.read()
        b64 = base64.b64encode(content).decode("utf-8")
        image_data = f"data:image/jpeg;base64,{b64}"

        vision = ChatOpenAI(model="gpt-4o", temperature=0)
        summary_prompt = [
            {"role": "user", "content": [
                {"type": "text",
                 "text": "Describe this frequency-response graph for an acoustic guitar: extract main peaks (Hz), dips, overall slope, and tonal implications."},
                {"type": "image_url", "image_url": {"url": image_data}}
            ]}
        ]
        vision_result = vision.invoke(summary_prompt)
        image_summary = vision_result.content.strip()
        print("🖼️ Image summary:", image_summary[:200])

        docs = retriever.get_relevant_documents(image_summary)
        context = synthesize_context(docs)

        interpret = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
        interpret_prompt = f"""
        You are Acustica, analyzing an acoustic-guitar frequency-response spectrum.

        User comment (optional): {comment or "—"}
        Image description:
        {image_summary}

        Context from corpus:
        {context}

        Provide one concise, physics-based interpretation (5-8 sentences) that
        explains the resonant behavior, identifies likely T(1,1) modes,
        and relates tonal balance or projection.
        """
        final = interpret.invoke(interpret_prompt)
        return {"analysis": final.content.strip()}

    except Exception as e:
        print("❌ Image analysis error:", e)
        return {"error": str(e)}
