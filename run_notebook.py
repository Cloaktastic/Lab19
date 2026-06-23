import json
import os
import sys
import re
import io
import time
import base64
import traceback
import numpy as np

from dotenv import load_dotenv
load_dotenv()
import os

# Load notebook
notebook_path = r"d:\VScode\Lab19\graphrag_lab.ipynb"
with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Shared execution context
exec_globals = {}

# Keep track of token usage and execution times
token_log = {
    "llm_prompt_tokens": 0,
    "llm_completion_tokens": 0,
    "embed_tokens": 0,
    "llm_calls": 0,
    "embed_calls": 0,
}

# Pricing for gpt-4o-mini and text-embedding-3-small (as of 2026)
# gpt-4o-mini: $0.150 per 1M input tokens, $0.600 per 1M output tokens
# text-embedding-3-small: $0.020 per 1M tokens
PRICING = {
    "gpt-4o-mini-input": 0.15 / 1_000_000,
    "gpt-4o-mini-output": 0.60 / 1_000_000,
    "embedding": 0.02 / 1_000_000
}

# Custom llm_complete with token tracking
def tracked_llm_complete(prompt, system=None, temperature=0.0, max_tokens=1200):
    from openai import OpenAI
    client = OpenAI()
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens
    )
    
    token_log["llm_prompt_tokens"] += resp.usage.prompt_tokens
    token_log["llm_completion_tokens"] += resp.usage.completion_tokens
    token_log["llm_calls"] += 1
    
    return resp.choices[0].message.content

# Custom embed with token tracking
def tracked_embed(texts):
    if isinstance(texts, str):
        texts = [texts]
    from openai import OpenAI
    client = OpenAI()
    resp = client.embeddings.create(model="text-embedding-3-small", input=texts)
    
    token_log["embed_tokens"] += resp.usage.prompt_tokens
    token_log["embed_calls"] += 1
    
    return [np.array(d.embedding, dtype=float) for d in resp.data]

# List of 20 benchmark questions
BENCHMARK_QUESTIONS = [
    "How does the electric vehicle sector connect to charging infrastructure and government policy?",
    "Which cities are mentioned as having zero-emission vehicle (ZEV) regulations, and what is their combined electric vehicle share?",
    "How do consumer incentives for EVs vary across U.S. metropolitan areas?",
    "What is the relationship between public charging availability and EV uptake share in top metropolitan areas?",
    "What are the common utility company actions to promote electric vehicles mentioned in the text?",
    "How does workplace charging availability compare to public charging availability in terms of EV adoption rates?",
    "What states or cities have the highest consumer incentives ranging from $1,500 to $5,500, and what form do these incentives take?",
    "How do U.S. zero-emission vehicle (ZEV) regulations impact the number of electric vehicle models available in those states?",
    "What role do HOV lane access, toll reductions, and free parking play in promoting electric vehicle uptake?",
    "How has the U.S. electric vehicle market share grown from 2010 to 2020?",
    "Which metropolitan areas had the greatest electric vehicle uptake in 2020, and how many promotion actions did they implement?",
    "What are the key differences in EV market success between states with and without ZEV regulations?",
    "What percentage of U.S. electric vehicle sales were states with ZEV regulations responsible for in 2020?",
    "How does charging infrastructure availability in top-ten markets compare to what half of the U.S. population has access to?",
    "Who are the authors of the study evaluating electric vehicle market growth across U.S. cities published on September 14, 2021?",
    "How many public chargers per million population did the top ten metropolitan areas average in 2020?",
    "What is the average electric vehicle share for states without ZEV regulations?",
    "How does the U.S. electric vehicle market size in 2018-2020 compare to the market size in 2010?",
    "Which specific fee reductions are listed as consumer incentives for EV market development?",
    "What are the key policy recommendations for accelerating electric vehicle market growth in metropolitan areas?"
]

# Run cells
print("Starting execution of notebook cells...")
for idx, cell in enumerate(nb["cells"]):
    cell_type = cell["cell_type"]
    if cell_type != "code":
        continue
        
    source_lines = cell["source"]
    source_code = "".join(source_lines).strip()
    
    if not source_code:
        continue
        
    print(f"\n--- Running Cell {idx} ---")
    
    # Apply modifications
    # 1. Skip pip install, apt install, and ollama serve commands
    if "%pip install" in source_code or "!sudo apt-get" in source_code or "!curl" in source_code or "!nohup ollama" in source_code or "!ollama pull" in source_code:
        print("Skipping package installation/Ollama setup cell.")
        cell["outputs"] = []
        cell["execution_count"] = idx
        continue
        
    # 2. Modify CONFIG cell (Cell 5)
    if "LLM_PROVIDER" in source_code and "OLLAMA_MODEL" in source_code:
        print("Modifying CONFIG parameters...")
        source_code = source_code.replace('LLM_PROVIDER       = "ollama"', 'LLM_PROVIDER       = "openai"')
        source_code = source_code.replace('LLM_PROVIDER       = "gemini"', 'LLM_PROVIDER       = "openai"')
        source_code = source_code.replace('EXTRACTION_BACKEND = "prompt"', 'EXTRACTION_BACKEND = "langextract"')
        source_code = "from dotenv import load_dotenv\nload_dotenv()\n" + source_code
        cell["source"] = [line + "\n" for line in source_code.splitlines()]
        
    # 3. Modify load_documents / chunking cell (Cell 10)
    if "!unzip" in source_code or "/content/dataset" in source_code:
        print("Modifying data directory paths for local execution...")
        source_code = source_code.replace('!unzip -q -o /content/dataset.zip -d /content/', '# Skipped zip extraction')
        source_code = source_code.replace('DATASET_DIR = "/content/dataset"', 'DATASET_DIR = "dataset"')
        cell["source"] = [line + "\n" for line in source_code.splitlines()]
        
    # 4. Modify Provider-agnostic wrappers (Cell 6) to inject token tracking and mock google.colab
    if "def llm_complete" in source_code:
        print("Modifying wrappers and mocking google.colab...")
        # Inject mock userdata and google.colab
        mock_setup = """import sys
class MockUserdata:
    def get(self, key):
        return os.environ.get(key)
userdata = MockUserdata()
sys.modules['google.colab'] = type('sys', (), {'userdata': userdata})
"""
        source_code = mock_setup + "\n" + source_code
        cell["source"] = [line + "\n" for line in source_code.splitlines()]
        
    # 5. Define LX_PROMPT and LX_EXAMPLES before langextract extraction
    if "import langextract as lx" in source_code and "def parse_json_triples" in source_code:
        print("Injecting LX_PROMPT and LX_EXAMPLES...")
        lx_definitions = """import langextract as lx
LX_PROMPT = \"\"\"
Extract all entities and their relationships.
The extraction classes should be:
- entity: representing a named entity (e.g. company, person, sector, technology, place, product, metric). Attribute: "type".
- relationship: representing a relation between a subject and an object. Attributes: "subject" (subject entity name), "relation" (relation type, e.g. operates_in, invests_in, competes_with, located_in, produces, reported), "object" (object entity name), "subject_type" (type of subject), "object_type" (type of object).
\"\"\"

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
"""
        source_code = lx_definitions + "\n" + source_code
        cell["source"] = [line + "\n" for line in source_code.splitlines()]

    # 6. Intercept matplotlib show in cell 21 to save it
    if "plt.show()" in source_code:
        print("Modifying plot cell to save graph...")
        source_code = source_code.replace("plt.show()", "plt.savefig('knowledge_graph.png', dpi=300)\nplt.show()")
        cell["source"] = [line + "\n" for line in source_code.splitlines()]
        
    # 7. Modify questions comparison cell (Cell 33) to run our 20 questions
    if "questions = [" in source_code and "How does the electric vehicle sector connect" in source_code:
        print("Replacing question set with 20 benchmark questions...")
        # Replace the list of questions
        source_code = re.sub(r"questions = \[[^\]]*\]", "questions = " + repr(BENCHMARK_QUESTIONS), source_code, flags=re.DOTALL)
        cell["source"] = [line + "\n" for line in source_code.splitlines()]
        
    # Capture stdout and stderr
    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = stdout_buf
    sys.stderr = stderr_buf
    
    start_time = time.time()
    success = True
    
    try:
        # We inject tracked wrapper overrides
        if "llm_complete" in exec_globals:
            exec_globals["llm_complete"] = tracked_llm_complete
        if "embed" in exec_globals:
            exec_globals["embed"] = tracked_embed
            
        exec(source_code, exec_globals)
        
        # After cell 6 is evaluated, inject wrappers directly
        if "llm_complete" in exec_globals and exec_globals["llm_complete"] != tracked_llm_complete:
            exec_globals["llm_complete"] = tracked_llm_complete
        if "embed" in exec_globals and exec_globals["embed"] != tracked_embed:
            exec_globals["embed"] = tracked_embed
            
    except Exception as e:
        success = False
        traceback.print_exc(file=sys.stderr)
        print(f"CELL {idx} FAILED: {e}", file=sys.stderr)
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        
    elapsed = time.time() - start_time
    print(f"Cell finished in {elapsed:.2f} seconds. Success={success}")
    
    # Store outputs in Jupyter Cell format
    cell_outputs = []
    
    stdout_str = stdout_buf.getvalue()
    if stdout_str:
        cell_outputs.append({
            "output_type": "stream",
            "name": "stdout",
            "text": [line + "\n" for line in stdout_str.splitlines()]
        })
        # Print stdout to console too
        print("STDOUT:")
        print(stdout_str)
        
    stderr_str = stderr_buf.getvalue()
    if stderr_str:
        cell_outputs.append({
            "output_type": "stream",
            "name": "stderr",
            "text": [line + "\n" for line in stderr_str.splitlines()]
        })
        print("STDERR:")
        print(stderr_str)
        
    # If cell generated a plot, read it and add as image display data
    if "plt.savefig" in source_code and os.path.exists("knowledge_graph.png"):
        with open("knowledge_graph.png", "rb") as img_f:
            b64_img = base64.b64encode(img_f.read()).decode("utf-8")
        cell_outputs.append({
            "output_type": "display_data",
            "data": {
                "image/png": b64_img,
                "text/plain": ["<Figure size 1300x900 with 1 Axes>"]
            },
            "metadata": {}
        })
        
    cell["outputs"] = cell_outputs
    cell["execution_count"] = idx
    
    if not success:
        print("Aborting notebook execution due to cell failure.")
        break

# Save executed notebook
with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("\nExecuted notebook saved successfully.")

# Write the final python code file (.py)
py_path = r"d:\VScode\Lab19\graphrag_lab.py"
with open(py_path, "w", encoding="utf-8") as f:
    for idx, cell in enumerate(nb["cells"]):
        cell_type = cell["cell_type"]
        source_code = "".join(cell["source"])
        f.write(f"\n# ==========================================\n")
        f.write(f"# CELL {idx} ({cell_type})\n")
        f.write(f"# ==========================================\n")
        if cell_type == "code":
            f.write(source_code)
            f.write("\n")
        else:
            for line in source_code.splitlines():
                f.write(f"# {line}\n")

print(f"Source code exported successfully to: {py_path}")

# Write token usage / cost report
print("\n=== TOKEN LOG SUMMARY ===")
for k, v in token_log.items():
    print(f"{k}: {v}")

# Cost calculations
cost_input = token_log["llm_prompt_tokens"] * PRICING["gpt-4o-mini-input"]
cost_output = token_log["llm_completion_tokens"] * PRICING["gpt-4o-mini-output"]
cost_embed = token_log["embed_tokens"] * PRICING["embedding"]
total_cost = cost_input + cost_output + cost_embed

print(f"Calculated Input LLM Cost: ${cost_input:.5f}")
print(f"Calculated Output LLM Cost: ${cost_output:.5f}")
print(f"Calculated Embedding Cost: ${cost_embed:.5f}")
print(f"Total API Cost: ${total_cost:.5f}")

# Save cost summary to a file
with open("cost_summary.json", "w", encoding="utf-8") as f:
    json.dump({
        "token_log": token_log,
        "pricing": PRICING,
        "costs": {
            "input_llm_cost": cost_input,
            "output_llm_cost": cost_output,
            "embedding_cost": cost_embed,
            "total_cost": total_cost
        }
    }, f, indent=4)
