# ───────────────────────────────────────────────
# Acustica Brain — Embedder (v2, corpus only)
# Builds a Chroma vectorstore from verified text files
# Giuliano Nicoletti — December 2025
# ───────────────────────────────────────────────

from pathlib import Path
from dotenv import load_dotenv
import os
import shutil

# LangChain (stable with 0.3.27 stack)
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

# ───────────────────────────────────────────────
# 1. Paths & Environment
# ───────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
VECTOR_DIR = BASE_DIR / "vectorstore"
COLLECTION_NAME = "acustica_corpus_v2"

# Load API key
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Fallback: read from Acustica_API_Key.txt if .env missing
if not api_key:
    key_file = BASE_DIR / "Acustica_API_Key.txt"
    if key_file.exists():
        with open(key_file, "r", encoding="utf-8") as f:
            api_key = f.read().strip()
        os.environ["OPENAI_API_KEY"] = api_key

if not api_key:
    raise EnvironmentError("❌ Missing OPENAI_API_KEY in .env or Acustica_API_Key.txt")

# ───────────────────────────────────────────────
# 2. Collect text files (only clean corpus)
# ───────────────────────────────────────────────
def collect_text_files():
    include_folders = ["clean_verified", "brand"]
    text_files = []
    for folder in include_folders:
        folder_path = DATA_DIR / folder
        if folder_path.exists():
            text_files.extend(folder_path.glob("*.txt"))
    return text_files

text_files = collect_text_files()
if not text_files:
    raise FileNotFoundError("❌ No .txt files found in clean_verified or brand folders")

print(f"[INFO] Found {len(text_files)} text files to embed")

# ───────────────────────────────────────────────
# 3. Load and split documents
# ───────────────────────────────────────────────
docs = []
for file in text_files:
    loader = TextLoader(str(file), encoding="utf-8")
    doc = loader.load()
    for d in doc:
        d.metadata["source"] = file.parent.name + "/" + file.name
    docs.extend(doc)

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
splits = splitter.split_documents(docs)

print(f"[INFO] Created {len(splits)} chunks from {len(text_files)} files")

# ───────────────────────────────────────────────
# 4. Backup previous vectorstore (if any)
# ───────────────────────────────────────────────
backup_dir = VECTOR_DIR / f"{COLLECTION_NAME}_backup"
current_dir = VECTOR_DIR / COLLECTION_NAME

if current_dir.exists():
    print(f"[BACKUP] Previous collection found. Saving backup to {backup_dir}")
    if backup_dir.exists():
        shutil.rmtree(backup_dir)
    shutil.move(str(current_dir), str(backup_dir))
else:
    print("[BACKUP] No previous collection found — fresh start")

# ───────────────────────────────────────────────
# 5. Embedding & Vectorstore
# ───────────────────────────────────────────────
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vectorstore = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=str(VECTOR_DIR)
)

print(f"[INFO] Creating new collection: {COLLECTION_NAME}")

# Add new chunks in batches
batch_size = 128
for i in range(0, len(splits), batch_size):
    batch = splits[i:i + batch_size]
    vectorstore.add_documents(batch)
    print(f"[UPsert] Added {len(batch)} (total {i + len(batch)})")

print("──────────────────────────────────────────────")
print(f"[DONE] Embedded chunks: {len(splits)}")
print(f"[STORE] Collection: {COLLECTION_NAME}")
print(f"[STORE] Path: {VECTOR_DIR}")
print("──────────────────────────────────────────────")
print("✅ Ready for retrieval (Phase 3)")
