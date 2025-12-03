# ───────────────────────────────────────────────
# Acustica Brain — Phase 4 Validator (Matches Retriever, UTF-8-safe)
# ───────────────────────────────────────────────
# Mirrors retriever.py configuration:
#   • collection_name = "acustica_corpus_v1"
#   • embedding model  = "text-embedding-3-small"
#   • same vectorstore path
#   • same MMR-style retrieval (k=10)
# Exports validation_log.csv encoded as UTF-8-SIG to
# preserve special characters on Windows.
# ───────────────────────────────────────────────

from pathlib import Path
from dotenv import load_dotenv
import os
import pandas as pd
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

# ───────────────────────────────────────────────
# 1. Setup
# ───────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
VECTOR_DIR = BASE_DIR / "vectorstore"
COLLECTION_NAME = "acustica_corpus_v1"

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    key_file = BASE_DIR / "Acustica_API_Key.txt"
    if key_file.exists():
        with open(key_file, "r", encoding="utf-8") as f:
            api_key = f.read().strip()
            os.environ["OPENAI_API_KEY"] = api_key
    else:
        raise EnvironmentError("❌ No API key found (.env or Acustica_API_Key.txt)")

# ───────────────────────────────────────────────
# 2. Connect to vectorstore
# ───────────────────────────────────────────────
print("🔗 Connecting to vectorstore...")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key)
db = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=str(VECTOR_DIR),
)
print(f"✅ Connected to collection: {COLLECTION_NAME}\n")

# ───────────────────────────────────────────────
# 3. Sample queries
# ───────────────────────────────────────────────
sample_queries = [
    "main resonance modes of acoustic guitar",
    "bridge torque definition",
    "mechanical impedance of the soundboard",
    "timbre physical characterization",
    "air cavity Helmholtz mode formula",
    "string plucking force components",
    "difference between top and back mobility",
    "coupling between T(1,1)₁ and T(1,1)₂",
    "longitudinal vs transversal string forces",
    "modal mass estimation method",
    "tonewood damping tanδ meaning",
    "radiation efficiency of guitar body",
    "bridge rotation stiffness",
    "modal tuning strategy",
    "acoustic monopole and dipole components",
    "measurement of FRF with accelerometer",
    "effect of bridge pin height on break angle",
    "string tension vs scale length",
    "back plate resonance control",
    "air–top coupling in low frequency range",
]

# ───────────────────────────────────────────────
# 4. Retrieve documents
# ───────────────────────────────────────────────
print("🎸  Running validation queries...\n")

rows = []
for q in sample_queries:
    docs_scores = db.similarity_search_with_score(q, k=10)
    for doc, score in docs_scores:
        src = doc.metadata.get("source", "")
        preview = doc.page_content[:500].replace("\n", " ")
        conf = round((1 - score) * 100, 1)
        rows.append({
            "query": q,
            "source": src,
            "confidence_%": conf,
            "chunk_preview": preview
        })

# ───────────────────────────────────────────────
# 5. Export CSV (UTF-8-SIG encoding)
# ───────────────────────────────────────────────
df = pd.DataFrame(rows)
csv_path = BASE_DIR / "validation_log.csv"
df.to_csv(csv_path, index=False, encoding="utf-8-sig")

if df.empty:
    print("⚠️  No results found — check collection name or path.")
else:
    print(f"✅ Retrieved {len(df)} rows.")
    print(f"[OK] Exported to: {csv_path}")
    print("👉 Add columns: valid (Y/N) | notes")

print("──────────────────────────────────────────────")
