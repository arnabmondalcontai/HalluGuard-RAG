# GitHub Copilot Chat — Build Prompts for HalluGuard
### (RAG Hallucination Detection & Mitigation Project)

## How to use this file

GitHub Copilot Chat works best with **focused, sequential prompts** rather than one giant mega-prompt (unlike Windsurf/Cascade). Paste each phase below into Copilot Chat **one at a time**, in order, inside VS Code. Use `@workspace` at the start of each prompt so Copilot has context of your whole project folder once files exist.

**Setup before you start:**
1. Create a folder called `halluguard` and open it in VS Code
2. Open Copilot Chat (`Ctrl+Shift+I` or the chat icon in the sidebar)
3. Make sure you're in **Agent mode** (not just "Ask") so Copilot can create/edit files directly — select it from the mode dropdown at the bottom of the chat panel
4. Paste Phase 1 below, let it finish, review the files it creates, then paste Phase 2, and so on

---

## PHASE 1 — Project Scaffold

```
Create a complete project scaffold for a Python research project called HalluGuard — 
a real-time hallucination detection and mitigation system for RAG (Retrieval-Augmented 
Generation) pipelines.

Create this exact directory structure with empty __init__.py files where needed:

halluguard/
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
├── src/
│   ├── rag_pipeline/ (retriever.py, generator.py, pipeline.py)
│   ├── detection/ (nli_detector.py, embedding_detector.py, llm_judge.py, ensemble.py)
│   ├── mitigation/ (query_rewriter.py, reranker.py, self_reflection.py)
│   ├── evaluation/ (metrics.py, benchmark.py)
│   └── utils/ (data_loader.py, logger.py)
├── notebooks/
├── experiments/configs/
├── experiments/results/
├── tests/
└── app/ (app.py, api.py)

requirements.txt must include exactly these packages with these versions:
langchain==0.2.6
langchain-community==0.2.6
chromadb==0.5.0
sentence-transformers==3.0.1
faiss-cpu==1.8.0
groq==0.9.0
huggingface_hub==0.23.4
transformers==4.42.3
torch==2.3.1
scikit-learn==1.5.0
scipy==1.13.1
ragas==0.1.14
datasets==2.20.0
evaluate==0.4.2
wandb==0.17.4
pyyaml==6.0.1
gradio==4.37.2
fastapi==0.111.0
uvicorn==0.30.1
python-dotenv==1.0.1
tqdm==4.66.4
pandas==2.2.2
numpy==1.26.4
rank_bm25==0.2.2
nltk==3.8.1

.env.example should list: GROQ_API_KEY, HUGGINGFACE_TOKEN, WANDB_API_KEY

.gitignore should exclude: .env, __pycache__/, chroma_db/, *.pyc, .venv/, models/, 
experiments/results/*.json, .ipynb_checkpoints/

Create a basic README.md with the project title "HalluGuard: Real-Time Hallucination 
Detection and Mitigation for RAG Systems" and placeholder sections for Abstract, 
Architecture, Quick Start, Results, and Citation.

Do not implement any logic yet — just create the structure and config files.
```

---

## PHASE 2 — RAG Pipeline (Retriever + Generator)

```
@workspace Now implement the core RAG pipeline in src/rag_pipeline/.

In retriever.py, implement a HybridRetriever class:
- Dense retrieval using sentence-transformers "all-MiniLM-L6-v2" embeddings stored in ChromaDB
- Sparse retrieval using BM25 (rank_bm25 library)
- A retrieve(query: str, top_k: int = 5) -> List[dict] method that merges dense and sparse 
  results using Reciprocal Rank Fusion (RRF) and returns ranked chunks with content, source, 
  chunk_id, and score
- An add_documents(docs: List[str], metadatas: List[dict]) method
- Persist ChromaDB to ./chroma_db/

In generator.py, implement a LLMGenerator class:
- Uses the Groq API (groq Python SDK), model "llama3-8b-8192", temperature 0.1 by default
- A generate(query: str, context_chunks: List[str], system_prompt: str = None) -> str method
- Default system prompt: "Answer ONLY based on the provided context. If the answer is not 
  in the context, say 'I cannot find this in the provided documents.' Always cite which 
  part of the context supports your answer."
- Retry logic with exponential backoff (3 retries) for API failures
- Log token usage per call using the logging module (not print)

In pipeline.py, implement a RAGPipeline class that composes HybridRetriever + LLMGenerator.
A run(query: str) -> dict method should return: query, retrieved_chunks, raw_response, 
and latency_ms. Don't add hallucination detection yet — that comes in the next phase.

Use type hints and Google-style docstrings on every function. Use logging, never print().
```

---

## PHASE 3 — Hallucination Detection Module

```
@workspace Now implement the hallucination detection module in src/detection/.

In nli_detector.py, implement NLIDetector:
- Loads "cross-encoder/nli-deberta-v3-base" from HuggingFace (cache the model, load once)
- score(response: str, context_chunks: List[str]) -> float method:
  - Tokenize response into sentences using nltk.sent_tokenize
  - For each sentence, run NLI with premise = concatenated context, hypothesis = sentence
  - Return 1 - mean(entailment_probability) as the hallucination score (0=faithful, 1=hallucinated)
  - Batch sentences for efficiency (batch_size=8)

In embedding_detector.py, implement EmbeddingDetector:
- Uses "sentence-transformers/all-mpnet-base-v2"
- score(response, context_chunks) -> float: embed response and chunks, return 
  1 - max_cosine_similarity
- Also add sentence_level_score() returning per-sentence scores

In llm_judge.py, implement LLMJudge:
- Uses Groq API to ask the LLM to rate faithfulness 1-5 with reasoning, parses a JSON response 
  like {"faithfulness_score": int, "reasoning": str, "problematic_claims": List[str]}
- Include 2-3 few-shot examples of faithful vs hallucinated responses in the prompt
- Return normalized score 0.0-1.0

In ensemble.py, implement EnsembleDetector:
- Combines all three detectors with weights {"nli": 0.5, "embedding": 0.2, "llm_judge": 0.3}
- score(query, response, context_chunks) -> dict returns final_score, individual scores, 
  is_hallucinated (bool), and threshold (0.5 as a class constant)

Write all code with type hints, docstrings, and proper error handling (catch exceptions, 
log them, return safe fallback values).
```

---

## PHASE 4 — Mitigation Module (your novelty — confidence-adaptive routing)

```
@workspace Now implement the mitigation module in src/mitigation/. This is the core novel 
contribution of my thesis: a confidence-gated mitigation router.

In query_rewriter.py, implement QueryRewriter using Groq API:
- rewrite(original_query, failed_response, problematic_claims) -> List[str] returns 3 
  alternative, more specific phrasings of the query

In reranker.py, implement ChunkReranker:
- Loads "cross-encoder/ms-marco-MiniLM-L-6-v2"
- rerank(query, chunks, top_k=3) -> List[str] returns the most relevant chunks re-scored

In self_reflection.py, implement SelfReflectionMitigator:
- mitigate(query, hallucinated_response, context_chunks, detector, generator, max_iterations=3)
- Loop: ask the LLM which claims are unsupported, ask it to rewrite removing them, 
  re-score with the detector, stop when score < threshold or max_iterations reached
- Return final_response, iterations, final_score, improvement

Now create a new file src/mitigation/router.py implementing ConfidenceAdaptiveRouter — 
this is my key novelty. It should:
- Accept the ensemble detector's hallucination_score
- Route to a strategy tier based on confidence:
  - 0.5–0.65 (low confidence hallucination): use QueryRewriter + re-retrieve + re-generate
  - 0.65–0.80 (medium confidence): use ChunkReranker on existing retrieved chunks + re-generate
  - >0.80 (high confidence): use SelfReflectionMitigator
- Return a dict with strategy_used, original_score, final_score, and improvement
- This routing logic should be implemented as a clean, well-documented function since I will 
  cite this exact design in my IEEE paper as a novel contribution distinct from prior work 
  (Self-RAG uses one fixed strategy, CRAG uses one fixed strategy — mine adapts by confidence tier)

Update src/rag_pipeline/pipeline.py to integrate EnsembleDetector and ConfidenceAdaptiveRouter: 
if hallucination_score > 0.5, automatically call the router and return the mitigated response.
```

---

## PHASE 5 — Evaluation & Datasets

```
@workspace Implement the evaluation module in src/evaluation/ and dataset utilities in 
src/utils/data_loader.py.

In data_loader.py:
- load_halueval(split="test") -> List[dict]: load from HuggingFace dataset "pminervini/HaluEval"
- load_truthfulqa() -> List[dict]: load from HuggingFace "truthful_qa"
- All loaders return schema: {"query": str, "context": List[str], "response": str, "label": int}

In metrics.py implement:
- faithfulness_score(), hallucination_detection_auroc(), hallucination_detection_f1(), 
  latency_stats() (mean, p50, p95, p99), mitigation_improvement_rate()

In benchmark.py implement a Benchmark class:
- run_on_halueval(split="test", n_samples=500): run the full pipeline + detector + router 
  on samples, compute all metrics
- run_ablation_study(): compare NLI-only vs Embedding-only vs LLM-judge-only vs Ensemble, 
  and compare with-router vs without-router (fixed single strategy) to demonstrate the 
  benefit of confidence-adaptive routing
- generate_results_table() -> pandas DataFrame formatted for a paper table
- Save results to experiments/results/ as timestamped JSON

Use tqdm progress bars for all dataset loops.
```

---

## PHASE 6 — Tests

```
@workspace Write pytest tests in tests/.

In test_detector.py:
- test_nli_detector_faithful(): a known faithful (query, context, response) triple should score < 0.3
- test_nli_detector_hallucinated(): a known hallucinated triple should score > 0.7
- test_ensemble_weighted_average(): mock the three detectors and verify the weighted average math
- test_threshold_detection(): score 0.6 with threshold 0.5 should set is_hallucinated=True

In test_pipeline.py:
- test_pipeline_returns_required_keys(): run() output must contain all expected keys
- test_router_triggers_on_high_score(): mock detector to return 0.9, verify SelfReflectionMitigator 
  is called (not query rewriter or reranker)
- test_router_tier_selection(): verify scores of 0.55, 0.7, 0.85 route to the correct strategy each

Run pytest and fix any failures before stopping.
```

---

## PHASE 7 — Gradio Demo App

```
@workspace Build a Gradio demo in app/app.py using gr.Blocks() with three tabs:

Tab 1 "Live Demo": query input, knowledge base text input (with a default sample loaded), 
a "Run HalluGuard" button, and outputs showing: the response, a color-coded hallucination 
score bar (green <0.3, yellow 0.3-0.5, red >0.5), a table breaking down NLI/Embedding/LLM-judge 
scores, a badge showing which mitigation strategy was used (if any), and the response with 
hallucinated sentences highlighted in red if found.

Tab 2 "Benchmark Results": load and display results from experiments/results/ as a table 
and a bar chart (gr.BarPlot) comparing detector methods and with/without router.

Tab 3 "About": project description and a placeholder for the paper abstract.

Use gr.themes.Soft(). Include 3 clickable example queries.
```

---

## PHASE 8 — FastAPI + Deployment Files

```
@workspace Implement app/api.py as a FastAPI app with:
- POST /detect: body {query, context, response} → hallucination_score, is_hallucinated, breakdown
- POST /rag: body {query, documents} → full pipeline output including mitigation
- GET /health
Add CORS middleware, Pydantic models for all request/response schemas, and OpenAPI docs at /docs.

Then create a Dockerfile for Hugging Face Spaces deployment (Python 3.10-slim base, installs 
requirements.txt, runs app/app.py on port 7860 with GRADIO_SERVER_NAME=0.0.0.0), and an 
app/README.md with the Hugging Face Spaces YAML header (title: HalluGuard, sdk: gradio, 
sdk_version: 4.37.2, app_file: app/app.py).
```

---

## Tips specific to Copilot Chat

- **If a phase is too large and Copilot stops partway**, just say: `continue implementing the rest of this phase`
- **If Copilot creates a file with errors**, paste the error message back with: `fix this error: [paste error]`
- **Switch to Edit mode** for small tweaks to a single file instead of Agent mode (faster, more surgical)
- **After each phase**, run `pip install -r requirements.txt --break-system-packages` (or just `pip install -r requirements.txt` in a venv) and try running the relevant file to catch issues early — don't wait until the end
- Get your free **Groq API key** at console.groq.com before Phase 2, and put it in your `.env` file (copy `.env.example` to `.env` first)

