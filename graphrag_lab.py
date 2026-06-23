
# ==========================================
# CELL 0 (markdown)
# ==========================================
# # Lab Day 19 — Building a GraphRAG System
# 
# **Goal:** build a full **GraphRAG** pipeline from a raw text corpus and compare it
# against a plain **Flat RAG** baseline.
# 
# This notebook walks through the lab end-to-end:
# 
# 1. **Entity & Relation Extraction** — turn unstructured text into `(subject, relation, object)` triples with an LLM or **[LangExtract](https://github.com/google/langextract)** (few-shot, source-grounded extraction).
# 2. **Graph Construction** — deduplicate entities and build a knowledge graph (**NetworkX**, optionally mirrored to **Neo4j**).
# 3. **Indexing** — embed chunks into a vector index for retrieval.
# 4. **GraphRAG Querying** — link a question to graph entities, traverse a **multi-hop** subgraph, and answer from graph facts + supporting text.
# 5. **Flat RAG vs GraphRAG** — run the same questions through both and compare.
# 
# > **Runs offline by default** using **Ollama** + **NetworkX** (no API key, no database).
# > Switch to **OpenAI** and/or **Neo4j** by changing the single `CONFIG` cell below.

# ==========================================
# CELL 1 (markdown)
# ==========================================
# ## Part 0 — Setup & Configuration
# 
# **Before running:**
# 
# - *Offline path (default):* install [Ollama](https://ollama.com), then in a terminal:
#   ```
#   ollama serve
#   ollama pull llama3.1
#   ollama pull nomic-embed-text
#   ```
# - *OpenAI path:* set `LLM_PROVIDER = "openai"` in CONFIG and export `OPENAI_API_KEY`.
# - *LangExtract path (recommended):* set `EXTRACTION_BACKEND = "langextract"` in CONFIG — works with Ollama (offline) or OpenAI; uses few-shot examples for more consistent triples.
# - *Neo4j (optional):* set `GRAPH_BACKEND = "neo4j"` and fill the Neo4j credentials.

# ==========================================
# CELL 2 (code)
# ==========================================
%pip install -q networkx numpy pandas matplotlib requests tqdm google-generativeai neo4j langextract

# ==========================================
# CELL 3 (code)
# ==========================================
!sudo apt-get install -y zstd
!curl -fsSL https://ollama.com/install.sh | sh
!nohup ollama serve > /dev/null 2>&1 &
import time
time.sleep(3)
!ollama pull llama3.1
!ollama pull nomic-embed-text

# ==========================================
# CELL 4 (code)
# ==========================================
import os, re, json, glob, difflib, collections
import numpy as np
import pandas as pd
import requests
import networkx as nx
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

# ==========================================
# CELL 5 (code)
# ==========================================
import os
os.environ['OPENAI_API_KEY'] = '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'
import os
os.environ['OPENAI_API_KEY'] = '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'
# ====================== CONFIG ======================
LLM_PROVIDER       = "openai"        # "ollama" | "openai" | "gemini"
GRAPH_BACKEND      = "networkx"      # "networkx" | "neo4j"
EXTRACTION_BACKEND = "langextract"   # "langextract" | "prompt"

# --- Gemini (used when LLM_PROVIDER == "gemini") ---
# Hãy đảm bảo bạn đã đặt GOOGLE_API_KEY trong phần Secrets của Colab
GEMINI_MODEL       = "gemini-2.0-flash-exp"
GEMINI_EMBED_MODEL = "models/text-embedding-004"

# --- OpenAI (used when LLM_PROVIDER == "openai") ---
OPENAI_MODEL       = "gpt-4o-mini"
OPENAI_EMBED_MODEL = "text-embedding-3-small"

# --- Ollama ---
OLLAMA_HOST        = "http://localhost:11434"
OLLAMA_MODEL       = "llama3.1"
OLLAMA_EMBED_MODEL = "nomic-embed-text"

# --- Data / runtime ---
DATASET_DIR   = "dataset"
MAX_DOCS      = 12
CHUNK_SIZE    = 1200
CHUNK_OVERLAP = 150

print(f"LLM_PROVIDER={LLM_PROVIDER} | EXTRACTION_BACKEND={EXTRACTION_BACKEND} | GRAPH_BACKEND={GRAPH_BACKEND}")


# ==========================================
# CELL 6 (code)
# ==========================================
import sys
class MockUserdata:
    def get(self, key):
        return os.environ.get(key)
userdata = MockUserdata()
sys.modules['google.colab'] = type('sys', (), {'userdata': userdata})

import os
os.environ['OPENAI_API_KEY'] = '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'
class MockUserdata:
    def get(self, key):
        return os.environ.get(key)
userdata = MockUserdata()
sys.modules['google.colab'] = type('sys', (), {'userdata': userdata})

import os
os.environ['OPENAI_API_KEY'] = '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'
# ---- Provider-agnostic LLM + embedding wrappers ----
import google.generativeai as genai
from google.colab import userdata

def llm_complete(prompt, system=None, temperature=0.0, max_tokens=1200):
    if LLM_PROVIDER == "gemini":
        genai.configure(api_key=userdata.get('GOOGLE_API_KEY'))
        model = genai.GenerativeModel(
            model_name=GEMINI_MODEL,
            system_instruction=system
        )
        resp = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens
            )
        )
        return resp.text

    elif LLM_PROVIDER == "openai":
        from openai import OpenAI
        client = OpenAI()
        messages = []
        if system: messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        resp = client.chat.completions.create(
            model=OPENAI_MODEL, messages=messages,
            temperature=temperature, max_tokens=max_tokens)
        return resp.choices[0].message.content

    elif LLM_PROVIDER == "ollama":
        payload = {"model": OLLAMA_MODEL, "prompt": prompt, "stream": False,
                   "options": {"temperature": temperature, "num_predict": max_tokens}}
        if system: payload["system"] = system
        r = requests.post(OLLAMA_HOST + "/api/generate", json=payload, timeout=600)
        r.raise_for_status()
        return r.json()["response"]
    raise ValueError("Unknown LLM_PROVIDER: " + LLM_PROVIDER)

def embed(texts):
    if isinstance(texts, str):
        texts = [texts]

    if LLM_PROVIDER == "gemini":
        genai.configure(api_key=userdata.get('GOOGLE_API_KEY'))
        result = genai.embed_content(model=GEMINI_EMBED_MODEL, content=texts, task_type="retrieval_document")
        return [np.array(e, dtype=float) for e in result['embedding']]

    if LLM_PROVIDER == "openai":
        from openai import OpenAI
        client = OpenAI()
        resp = client.embeddings.create(model=OPENAI_EMBED_MODEL, input=texts)
        return [np.array(d.embedding, dtype=float) for d in resp.data]

    out = []
    for t in texts:
        r = requests.post(OLLAMA_HOST + "/api/embeddings",
                          json={"model": OLLAMA_EMBED_MODEL, "prompt": t}, timeout=600)
        r.raise_for_status()
        out.append(np.array(r.json()["embedding"], dtype=float))
    return out


# ==========================================
# CELL 7 (code)
# ==========================================
# Smoke test — verifies the chosen provider responds (safe to re-run).
try:
    print("LLM says:", llm_complete("Reply with exactly: OK", max_tokens=10).strip()[:50])
    print("Embedding dim:", len(embed("hello world")[0]))
except Exception as e:
    print("Provider not reachable yet:", e)
    print("Ollama:  run `ollama serve` and pull `llama3.1` + `nomic-embed-text`.")
    print("OpenAI:  set LLM_PROVIDER='openai' and export OPENAI_API_KEY.")

# ==========================================
# CELL 8 (markdown)
# ==========================================
# ## Part 1 — Concepts (the lab's research questions)
# 
# **1. Entity Extraction — how does the LLM tell an *entity* from an *attribute*?**
# We extract structured triples `(subject, relation, object)` with entity *types*.
# By default we use **[LangExtract](https://github.com/google/langextract)** — few-shot examples
# teach the schema, and each extraction can be traced back to its source span. The `prompt`
# backend is a simpler single-shot JSON alternative. Entities become *nodes*; relationships become *edges*.
# 
# **2. Graph Construction — why does deduplication matter?**
# The same real-world entity appears under many surface forms ("Tesla", "Tesla Inc.", "tesla").
# If we don't merge them, the graph fragments: facts about one entity scatter across several nodes,
# breaking traversal. We normalize names and fuzzy-merge near-duplicates into a single canonical node.
# 
# **3. Query Answering — graph traversal vs. plain vector search?**
# *Flat RAG* embeds the query and returns the top-k most *similar* chunks — great for "what does the text say
# about X", but blind to relationships that aren't co-located in one passage.
# *GraphRAG* links the query to entities and **traverses** the graph (multi-hop) to assemble connected facts —
# so it can answer "how is A connected to C *through* B" even when no single chunk states it.

# ==========================================
# CELL 9 (markdown)
# ==========================================
# ## Part 2 — Load & Chunk the Corpus
# 
# We read the `*.txt` files from `DATASET_DIR` and split each into overlapping character chunks.

# ==========================================
# CELL 10 (code)
# ==========================================
# Skipped zip extraction
DATASET_DIR = "dataset"

def load_documents(directory, max_docs=None):
    paths = sorted(glob.glob(os.path.join(directory, "*.txt")))
    if max_docs:
        paths = paths[:max_docs]
    docs = []
    for p in paths:
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            docs.append({"doc_id": os.path.basename(p), "text": f.read()})
    return docs

def chunk_text(text, size, overlap):
    text = text.strip()
    chunks, start = [], 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = max(0, end - overlap)
    return chunks


documents = load_documents(DATASET_DIR, MAX_DOCS)
chunks = []
for d in documents:
    for j, ctext in enumerate(chunk_text(d["text"], CHUNK_SIZE, CHUNK_OVERLAP)):
        chunks.append({"chunk_id": f"{d['doc_id']}::ch{j}",
                       "doc_id": d["doc_id"], "text": ctext})

print(f"Loaded {len(documents)} docs -> {len(chunks)} chunks")
print("\nExample chunk:\n", chunks[0]["text"][:300])


# ==========================================
# CELL 11 (markdown)
# ==========================================
# ## Part 3 — Entity & Relation Extraction
# 
# Two backends (set `EXTRACTION_BACKEND` in CONFIG):
# 
# | Backend | How it works | Best for |
# |---------|--------------|----------|
# | **`langextract`** (default) | Google's [LangExtract](https://github.com/google/langextract) library — few-shot examples, schema-guided output, source grounding | More consistent triples; same Ollama/OpenAI provider as the rest of the lab |
# | **`prompt`** | Single-shot JSON prompt to the LLM | Minimal dependencies; quick baseline |
# 
# Both produce the same `(subject, relation, object)` triple list consumed by graph construction below.

# ==========================================
# CELL 12 (code)
# ==========================================

LX_PROMPT = """
Extract all entities and their relationships.
The extraction classes should be:
- entity: representing a named entity (e.g. company, person, sector, technology, place, product, metric). Attribute: "type".
- relationship: representing a relation between a subject and an object. Attributes: "subject" (subject entity name), "relation" (relation type, e.g. operates_in, invests_in, competes_with, located_in, produces, reported), "object" (object entity name), "subject_type" (type of subject), "object_type" (type of object).
"""

LX_EXAMPLES = [
    lx.data.ExampleData(
        text="Tesla operates in the U.S. and produces electric vehicles.",
        extractions=[
            lx.data.Extraction(
                extraction_class="entity",
                extraction_text="Tesla",
                attributes={"type": "company"}
            ),
            lx.data.Extraction(
                extraction_class="entity",
                extraction_text="U.S.",
                attributes={"type": "place"}
            ),
            lx.data.Extraction(
                extraction_class="entity",
                extraction_text="electric vehicles",
                attributes={"type": "product"}
            ),
            lx.data.Extraction(
                extraction_class="relationship",
                extraction_text="Tesla operates in the U.S.",
                attributes={
                    "subject": "Tesla",
                    "subject_type": "company",
                    "relation": "operates_in",
                    "object": "U.S.",
                    "object_type": "place"
                }
            ),
            lx.data.Extraction(
                extraction_class="relationship",
                extraction_text="Tesla produces electric vehicles.",
                attributes={
                    "subject": "Tesla",
                    "subject_type": "company",
                    "relation": "produces",
                    "object": "electric vehicles",
                    "object_type": "product"
                }
            )
        ]
    )
]

import os
os.environ['OPENAI_API_KEY'] = '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'
import textwrap
import langextract as lx

# ... (parse_json_triples and extract_triples_prompt stay the same) ...
def parse_json_triples(raw):
    if not raw: return []
    text = raw.strip()
    text = re.sub(r"^```[a-zA-Z]*", "", text).strip().strip("`").strip()
    try:
        data = json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m: return []
        try: data = json.loads(m.group(0))
        except Exception: return []
    return data.get("triples", []) if isinstance(data, dict) else []

EXTRACT_SYSTEM = "You are an information-extraction engine that builds knowledge graphs from text."

def extract_triples_prompt(text):
    prompt = (
        "Extract the key ENTITIES and RELATIONSHIPS from the text below.\n"
        "Return ONLY valid JSON in this exact shape (no prose, no markdown):\n"
        '{"triples": [{"subject": "", "subject_type": "", "relation": "", "object": "", "object_type": ""}]}\n\n'
        "Guidelines:\n"
        "- subject/object must be specific named entities (company, sector, technology, product, place, person, metric).\n"
        "- relation is a short snake_case verb phrase: operates_in, invests_in, competes_with, located_in, produces, reported.\n"
        "- Use canonical, concise entity names. Skip vague pronouns.\n"
        "- Return at most 15 of the most important triples.\n\n"
        "Text:\n<<<\n" + text[:3500] + "\n>>>"
    )
    raw = llm_complete(prompt, system=EXTRACT_SYSTEM, max_tokens=1200)
    return parse_json_triples(raw)

# Updated model kwargs for LangExtract to support Gemini
def _langextract_model_kwargs():
    if LLM_PROVIDER == "ollama":
        return {"model_id": OLLAMA_MODEL, "model_url": OLLAMA_HOST}
    if LLM_PROVIDER == "openai":
        return {"model_id": OPENAI_MODEL}
    if LLM_PROVIDER == "gemini":
        return {"model_id": GEMINI_MODEL, "api_key": userdata.get('GOOGLE_API_KEY')}
    raise ValueError("Unknown LLM_PROVIDER: " + LLM_PROVIDER)

# ... (Rest of extraction functions remain the same) ...
def lx_extractions_to_triples(result):
    triples = []
    for ext in (result.extractions or []):
        if ext.extraction_class != "relationship": continue
        attrs = ext.attributes or {}
        s, o, r = attrs.get("subject"), attrs.get("object"), attrs.get("relation")
        if not (s and o and r): continue
        triples.append({
            "subject": str(s).strip(), "subject_type": str(attrs.get("subject_type", "")).strip(),
            "relation": str(r).strip(), "object": str(o).strip(),
            "object_type": str(attrs.get("object_type", "")).strip(),
        })
    return triples

def extract_triples_langextract(text):
    result = lx.extract(
        text_or_documents=text[:3500],
        prompt_description=LX_PROMPT,
        examples=LX_EXAMPLES,
        max_char_buffer=min(CHUNK_SIZE, 1200),
        **_langextract_model_kwargs(),
    )
    return lx_extractions_to_triples(result)

def extract_triples(text):
    if EXTRACTION_BACKEND == "langextract": return extract_triples_langextract(text)
    if EXTRACTION_BACKEND == "prompt": return extract_triples_prompt(text)
    raise ValueError("Unknown EXTRACTION_BACKEND: " + EXTRACTION_BACKEND)


# ==========================================
# CELL 13 (code)
# ==========================================
# Run extraction across all chunks (limited by MAX_DOCS). This is the slow step.
all_triples = []   # each: chunk_id, doc_id, subject, subject_type, relation, object, object_type

for ch in tqdm(chunks, desc=f"Extracting ({EXTRACTION_BACKEND})"):
    try:
        triples = extract_triples(ch["text"])
    except Exception:
        triples = []
    for t in triples:
        if not (t.get("subject") and t.get("object") and t.get("relation")):
            continue
        all_triples.append({
            "chunk_id": ch["chunk_id"], "doc_id": ch["doc_id"],
            "subject": str(t["subject"]).strip(),
            "subject_type": str(t.get("subject_type", "")).strip(),
            "relation": str(t["relation"]).strip(),
            "object": str(t["object"]).strip(),
            "object_type": str(t.get("object_type", "")).strip(),
        })

print(f"Extracted {len(all_triples)} raw triples from {len(chunks)} chunks")

# ==========================================
# CELL 14 (code)
# ==========================================
pd.DataFrame(all_triples).head(15)

# ==========================================
# CELL 15 (markdown)
# ==========================================
# ### (Optional) LangExtract source-grounding visualization
# 
# When `EXTRACTION_BACKEND == "langextract"`, re-run extraction on one chunk and generate
# an interactive HTML map of extractions back to source text spans.

# ==========================================
# CELL 16 (code)
# ==========================================
if EXTRACTION_BACKEND == "langextract":
    demo_text = chunks[0]["text"][:2500]
    lx_result = lx.extract(
        text_or_documents=demo_text,
        prompt_description=LX_PROMPT,
        examples=LX_EXAMPLES,
        max_char_buffer=min(CHUNK_SIZE, 1200),
        **_langextract_model_kwargs(),
    )
    lx.io.save_annotated_documents([lx_result], output_name="lx_demo.jsonl", output_dir=".")
    html = lx.visualize("lx_demo.jsonl")
    html_path = "lx_extraction_viz.html"
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html.data if hasattr(html, "data") else html)
    print(f"Saved interactive visualization -> {html_path}")
    print(f"Grounded extractions: {sum(1 for e in lx_result.extractions if e.char_interval)} / {len(lx_result.extractions)}")
else:
    print("Set EXTRACTION_BACKEND='langextract' to generate the HTML visualization.")

# ==========================================
# CELL 17 (markdown)
# ==========================================
# ## Part 4 — Graph Construction (with Deduplication)
# 
# We normalize entity names and fuzzy-merge near-duplicates into canonical nodes,
# then build a directed multigraph in **NetworkX**. Each edge remembers which chunk it came from
# (provenance), so GraphRAG can pull the supporting text later.

# ==========================================
# CELL 18 (code)
# ==========================================
def normalize(name):
    n = re.sub(r"\s+", " ", str(name)).strip()
    return n.strip(' .,:;"\'')

# Build a canonical alias map by fuzzy-merging surface forms.
raw_entities = sorted({normalize(t[k]) for t in all_triples for k in ("subject", "object")} - {""})

canonical, canon_list = {}, []
for e in raw_entities:
    match = difflib.get_close_matches(e.lower(), [c.lower() for c in canon_list], n=1, cutoff=0.92)
    if match:
        canonical[e] = next(c for c in canon_list if c.lower() == match[0])
    else:
        canon_list.append(e)
        canonical[e] = e

def canon(name):
    return canonical.get(normalize(name), normalize(name))

print(f"{len(raw_entities)} surface entities -> {len(canon_list)} canonical entities after dedup")

# ==========================================
# CELL 19 (code)
# ==========================================
G = nx.MultiDiGraph()
entity_chunks = {}   # canonical entity -> set of chunk_ids (provenance)
entity_type = {}

for t in all_triples:
    s, o = canon(t["subject"]), canon(t["object"])
    if not s or not o:
        continue
    if t["subject_type"]:
        entity_type.setdefault(s, t["subject_type"])
    if t["object_type"]:
        entity_type.setdefault(o, t["object_type"])
    G.add_node(s, type=entity_type.get(s, ""))
    G.add_node(o, type=entity_type.get(o, ""))
    G.add_edge(s, o, relation=t["relation"], chunk_id=t["chunk_id"], doc_id=t["doc_id"])
    entity_chunks.setdefault(s, set()).add(t["chunk_id"])
    entity_chunks.setdefault(o, set()).add(t["chunk_id"])

print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# ==========================================
# CELL 20 (code)
# ==========================================
print("Top entities by degree:")
for n, d in sorted(G.degree, key=lambda x: x[1], reverse=True)[:15]:
    print(f"  {d:3d}  {n}  ({G.nodes[n].get('type','')})")

# ==========================================
# CELL 21 (code)
# ==========================================
# Visualize the most-connected slice of the graph
top_nodes = [n for n, _ in sorted(G.degree, key=lambda x: x[1], reverse=True)[:25]]
H = G.subgraph(top_nodes)

plt.figure(figsize=(13, 9))
pos = nx.spring_layout(H, k=0.6, seed=42)
nx.draw_networkx_nodes(H, pos, node_size=900, node_color="#9ecae1")
nx.draw_networkx_labels(H, pos, font_size=8)
nx.draw_networkx_edges(H, pos, alpha=0.3, arrows=True)
nx.draw_networkx_edge_labels(
    H, pos, font_size=6,
    edge_labels={(u, v): d["relation"] for u, v, d in H.edges(data=True)})
plt.title("Knowledge Graph — top 25 entities by degree")
plt.axis("off"); plt.tight_layout(); plt.show()

# ==========================================
# CELL 22 (markdown)
# ==========================================
# ### (Optional) Mirror the graph into Neo4j
# 
# Runs only when `GRAPH_BACKEND == "neo4j"` and credentials are set — otherwise it is skipped,
# so the notebook still runs fully offline.

# ==========================================
# CELL 23 (code)
# ==========================================
if GRAPH_BACKEND == "neo4j":
    from neo4j import GraphDatabase
    driver = GraphDatabase.driver(NEO4J_URL, auth=(NEO4J_USER, NEO4J_PASSWORD))
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
        for n, data in G.nodes(data=True):
            session.run("MERGE (e:Entity {name:$name}) SET e.type=$type",
                        name=n, type=data.get("type", ""))
        for u, v, data in G.edges(data=True):
            session.run(
                "MATCH (a:Entity {name:$u}), (b:Entity {name:$v}) "
                "MERGE (a)-[r:REL {type:$rel}]->(b) SET r.doc_id=$doc",
                u=u, v=v, rel=data.get("relation", ""), doc=data.get("doc_id", ""))
    driver.close()
    print("Pushed graph to Neo4j at", NEO4J_URL)
else:
    print("GRAPH_BACKEND != 'neo4j' -> skipping. Set GRAPH_BACKEND='neo4j' + creds in CONFIG to enable.")

# ==========================================
# CELL 24 (markdown)
# ==========================================
# **Multi-hop traversal in Cypher** (run in the Neo4j Browser once the graph is loaded):
# 
# ```cypher
# // 2-hop neighbourhood around an entity
# MATCH path = (a:Entity {name: "Tesla"})-[*1..2]-(b:Entity)
# RETURN path LIMIT 50;
# ```

# ==========================================
# CELL 25 (markdown)
# ==========================================
# ## Part 5 — Vector Index (for the Flat RAG baseline)
# 
# A minimal cosine-similarity index over the chunk embeddings — this is what plain RAG retrieves from.

# ==========================================
# CELL 26 (code)
# ==========================================
class VectorIndex:
    def __init__(self):
        self.ids, self.meta, self.matrix = [], {}, None

    def build(self, items):
        vecs = []
        for it in tqdm(items, desc="Embedding"):
            vecs.append(embed(it["text"])[0])
            self.ids.append(it["chunk_id"])
            self.meta[it["chunk_id"]] = it
        m = np.vstack(vecs)
        self.matrix = m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-9)

    def search(self, query, k=4):
        q = embed(query)[0]
        q = q / (np.linalg.norm(q) + 1e-9)
        sims = self.matrix @ q
        order = np.argsort(-sims)[:k]
        return [(self.ids[i], float(sims[i]), self.meta[self.ids[i]]) for i in order]

vindex = VectorIndex()
vindex.build(chunks)
print("Vector index:", len(vindex.ids), "chunks, dim =", vindex.matrix.shape[1])

# ==========================================
# CELL 27 (markdown)
# ==========================================
# ## Part 6 — GraphRAG Querying (multi-hop)
# 
# The GraphRAG retrieval pipeline:
# 
# 1. **Entity linking** — find graph nodes mentioned in the question.
# 2. **Multi-hop traversal** — expand a k-hop subgraph around those seeds.
# 3. **Verbalize** the subgraph edges into facts + pull the **provenance chunks**.
# 4. **Answer** from graph facts *and* supporting text.

# ==========================================
# CELL 28 (code)
# ==========================================
def link_entities(query, max_seeds=5):
    ql = query.lower()
    seeds = [n for n in G.nodes if n.lower() in ql]
    if not seeds:                      # fuzzy fallback on query tokens
        names = list(G.nodes)
        for token in re.findall(r"[A-Za-z][A-Za-z0-9&.\- ]{2,}", query):
            m = difflib.get_close_matches(token.strip().lower(),
                                          [x.lower() for x in names], n=1, cutoff=0.8)
            if m:
                seeds += [x for x in names if x.lower() == m[0] and x not in seeds]
    return sorted(set(seeds), key=lambda n: G.degree(n), reverse=True)[:max_seeds]

def k_hop_subgraph(seeds, hops=2, max_nodes=40):
    nodes, frontier = set(seeds), set(seeds)
    for _ in range(hops):
        nxt = set()
        for n in frontier:
            nxt.update(G.successors(n)); nxt.update(G.predecessors(n))
        nodes.update(nxt); frontier = nxt
        if len(nodes) >= max_nodes:
            break
    return G.subgraph(list(nodes)[:max_nodes])

def verbalize(sg):
    lines = {f"({u}) -[{d.get('relation','related_to')}]-> ({v})"
             for u, v, d in sg.edges(data=True)}
    return "\n".join(sorted(lines))

def gather_chunks(sg, limit=6):
    cids, texts = [], []
    for n in sg.nodes:
        for cid in entity_chunks.get(n, []):
            if cid not in cids:
                cids.append(cid)
    by_id = {c["chunk_id"]: c for c in chunks}
    for cid in cids[:limit]:
        if cid in by_id:
            texts.append(f"[{cid}] " + by_id[cid]["text"][:600])
    return texts

def graphrag_answer(query, hops=2):
    seeds = link_entities(query)
    sg = k_hop_subgraph(seeds, hops=hops)
    facts = verbalize(sg)
    evidence = "\n\n".join(gather_chunks(sg))
    prompt = (
        "Answer the question using the KNOWLEDGE-GRAPH FACTS and SUPPORTING TEXT below. "
        "Reason across multiple connected facts (multi-hop) when needed. "
        "If the context is insufficient, say so.\n\n"
        "QUESTION: " + query + "\n\n"
        "KNOWLEDGE-GRAPH FACTS:\n" + (facts or "(none found)") + "\n\n"
        "SUPPORTING TEXT:\n" + (evidence or "(none)") + "\n\n"
        "ANSWER:"
    )
    return {"answer": llm_complete(prompt, max_tokens=500), "seeds": seeds,
            "n_facts": len(facts.splitlines()) if facts else 0}

# ==========================================
# CELL 29 (code)
# ==========================================
q = "How does the electric vehicle sector connect to charging infrastructure and government policy?"
res = graphrag_answer(q, hops=2)
print("Seed entities:", res["seeds"])
print("Graph facts used:", res["n_facts"])
print("\nGraphRAG answer:\n", res["answer"])

# ==========================================
# CELL 30 (markdown)
# ==========================================
# ## Part 7 — Flat RAG Baseline
# 
# Plain retrieval: embed the query, grab the top-k most similar chunks, answer from those passages only.

# ==========================================
# CELL 31 (code)
# ==========================================
def flat_rag_answer(query, k=4):
    hits = vindex.search(query, k=k)
    context = "\n\n".join(f"[{cid}] " + meta["text"][:700] for cid, _, meta in hits)
    prompt = (
        "Answer the question using ONLY the context passages below. "
        "If the answer is not in the context, say you don't have enough information.\n\n"
        "QUESTION: " + query + "\n\n"
        "CONTEXT:\n" + context + "\n\n"
        "ANSWER:"
    )
    return {"answer": llm_complete(prompt, max_tokens=500),
            "sources": [cid for cid, _, _ in hits]}

fr = flat_rag_answer(q)
print("Sources:", fr["sources"])
print("\nFlat RAG answer:\n", fr["answer"])

# ==========================================
# CELL 32 (markdown)
# ==========================================
# ## Part 8 — Flat RAG vs GraphRAG Comparison
# 
# Run the same question set through both pipelines. Look for questions that require *connecting*
# entities across documents — that is where GraphRAG's multi-hop traversal tends to win, while
# Flat RAG is limited to whatever a single similar passage happens to contain.

# ==========================================
# CELL 33 (code)
# ==========================================
questions = [
    "How does the electric vehicle sector connect to charging infrastructure and government policy?",
    "Which companies are mentioned in relation to EV market growth, and how are they connected?",
    "What financial trends or sentiments are associated with renewable energy investments?",
    "How do government incentives relate to EV adoption across regions?",
]

rows = []
for ques in tqdm(questions, desc="Comparing"):
    g = graphrag_answer(ques, hops=2)
    f = flat_rag_answer(ques, k=4)
    rows.append({"question": ques,
                 "graph_seeds": ", ".join(g["seeds"]) or "(none)",
                 "graphrag_answer": g["answer"],
                 "flatrag_answer": f["answer"]})

pd.set_option("display.max_colwidth", 350)
comparison = pd.DataFrame(rows)
comparison

# ==========================================
# CELL 34 (code)
# ==========================================
# Read the full answers side by side
for r in rows:
    print("=" * 90)
    print("Q:", r["question"])
    print("seeds:", r["graph_seeds"])
    print("\n-- GraphRAG --\n", r["graphrag_answer"])
    print("\n-- Flat RAG --\n", r["flatrag_answer"], "\n")

# ==========================================
# CELL 35 (markdown)
# ==========================================
# ## Part 9 — Conclusion & Takeaways
# 
# **What we built:** a complete GraphRAG pipeline — extraction (LangExtract or prompt) → dedup →
# knowledge graph → multi-hop retrieval → grounded answer — plus a Flat RAG baseline for comparison,
# all behind provider/backend switches (Ollama/OpenAI, NetworkX/Neo4j, LangExtract/prompt).
# 
# **When GraphRAG wins:** questions that require *connecting* facts spread across documents
# ("how is A related to C through B"). The graph encodes those relationships explicitly, so traversal
# surfaces them even when no single passage states the connection.
# 
# **When Flat RAG is enough:** direct lookups where the answer sits inside one passage — it is simpler
# and cheaper (no extraction step).
# 
# **Knobs to explore:**
# - Raise `MAX_DOCS` toward the full corpus and re-run.
# - Switch `EXTRACTION_BACKEND` between `langextract` (few-shot, grounded) and `prompt` (baseline).
# - Increase `hops` (2 → 3) for deeper multi-hop reasoning.
# - Tighten the extraction prompt / dedup `cutoff` for a cleaner graph.
# - Switch `LLM_PROVIDER` to `openai` for higher-quality extraction and answers.
# 
# **Interview-style questions to check understanding:**
# 1. Why does poor entity deduplication degrade GraphRAG more than Flat RAG?
# 2. How does multi-hop traversal answer questions that vector similarity alone cannot?
# 3. What are the cost/latency trade-offs of the extraction step versus plain embedding?
# 4. How would you evaluate GraphRAG vs Flat RAG quantitatively (e.g., an LLM judge or a labelled QA set)?
