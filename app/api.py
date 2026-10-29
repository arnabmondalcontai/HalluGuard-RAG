"""HalluGuard FastAPI REST API endpoint.

Provides REST endpoints for hallucination detection and full RAG pipeline
execution with CORS, rate limiting, request logging, and Pydantic schemas.
"""

import os
import sys
import time
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── Pydantic Schemas ────────────────────────────────────────────────────


class DetectRequest(BaseModel):
    """Request schema for the /detect endpoint."""

    query: str = Field(..., description="The original query.")
    context: List[str] = Field(..., description="List of context passages.")
    response: str = Field(..., description="The response to evaluate.")


class DetectResponse(BaseModel):
    """Response schema for the /detect endpoint."""

    hallucination_score: float = Field(
        ..., description="Hallucination score 0.0-1.0."
    )
    is_hallucinated: bool = Field(
        ..., description="Whether the response is considered hallucinated."
    )
    breakdown: Dict[str, Any] = Field(
        default_factory=dict, description="Per-detector score breakdown."
    )


class RAGRequest(BaseModel):
    """Request schema for the /rag endpoint."""

    query: str = Field(..., description="The user's question.")
    documents: List[str] = Field(
        ..., description="Knowledge base documents to retrieve from."
    )


class RAGResponse(BaseModel):
    """Response schema for the /rag endpoint."""

    query: str
    retrieved_chunks: List[str] = []
    raw_response: str = ""
    hallucination_score: float = 0.0
    hallucination_detected: bool = False
    final_response: str = ""
    mitigation_applied: str = "none"
    latency_ms: float = 0.0


class HealthResponse(BaseModel):
    """Response schema for the /health endpoint."""

    status: str = "ok"
    model_loaded: bool = False


# ── App Setup ───────────────────────────────────────────────────────────

app = FastAPI(
    title="HalluGuard API",
    description=(
        "Real-Time Hallucination Detection and Mitigation for RAG Systems. "
        "Provides endpoints for hallucination detection and full RAG pipeline execution."
    ),
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate limiting
try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.errors import RateLimitExceeded
    from slowapi.util import get_remote_address

    limiter = Limiter(key_func=get_remote_address)
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    RATE_LIMITING_ENABLED = True
except ImportError:
    logger.warning("slowapi not installed — rate limiting disabled.")
    RATE_LIMITING_ENABLED = False
    limiter = None


# Global pipeline
_pipeline = None


def get_pipeline():
    """Lazy-load the RAG pipeline."""
    global _pipeline
    if _pipeline is None:
        from dotenv import load_dotenv
        load_dotenv()

        from src.detection.ensemble import EnsembleDetector
        from src.mitigation.router import ConfidenceAdaptiveRouter
        from src.mitigation.self_reflection import SelfReflectionMitigator
        from src.rag_pipeline.generator import LLMGenerator
        from src.rag_pipeline.pipeline import RAGPipeline
        from src.rag_pipeline.retriever import HybridRetriever

        retriever = HybridRetriever()
        generator = LLMGenerator()
        detector = EnsembleDetector()
        mitigator = SelfReflectionMitigator()
        router = ConfidenceAdaptiveRouter(
            retriever=retriever,
            generator=generator,
            detector=detector,
            self_reflection=mitigator,
        )

        _pipeline = RAGPipeline(
            retriever=retriever,
            generator=generator,
            detector=detector,
            mitigator=mitigator,
            router=router,
        )
    return _pipeline


# ── Request logging middleware ──────────────────────────────────────────


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests and their response times."""
    start_time = time.time()
    response = await call_next(request)
    elapsed = (time.time() - start_time) * 1000

    logger.info(
        "%s %s — %d (%.2f ms)",
        request.method, request.url.path, response.status_code, elapsed,
    )

    return response


# ── Endpoints ───────────────────────────────────────────────────────────


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Check API health and model loading status.

    Returns:
        Health status and whether models are loaded.
    """
    return HealthResponse(
        status="ok",
        model_loaded=_pipeline is not None,
    )


@app.post("/detect", response_model=DetectResponse, tags=["Detection"])
async def detect_hallucination(request: DetectRequest):
    """Detect hallucination in a response given query and context.

    Runs the ensemble detector (NLI + Embedding + LLM Judge) on the provided
    response and returns a hallucination score with per-detector breakdown.

    Args:
        request: DetectRequest with query, context, and response.

    Returns:
        DetectResponse with hallucination score, detection flag, and breakdown.
    """
    pipeline = get_pipeline()

    result = pipeline.run_detection_only(
        query=request.query,
        response=request.response,
        context_chunks=request.context,
    )

    return DetectResponse(
        hallucination_score=result["hallucination_score"],
        is_hallucinated=result["is_hallucinated"],
        breakdown=result["breakdown"],
    )


@app.post("/rag", response_model=RAGResponse, tags=["RAG Pipeline"])
async def run_rag_pipeline(request: RAGRequest):
    """Run the full RAG pipeline with hallucination detection and mitigation.

    Indexes the provided documents, retrieves relevant context, generates
    a response, detects hallucination, and applies mitigation if needed.

    Args:
        request: RAGRequest with query and documents.

    Returns:
        Full pipeline output including response, scores, and mitigation details.
    """
    pipeline = get_pipeline()

    # Index documents
    pipeline.add_documents(
        docs=request.documents,
        metadatas=[
            {"source": "api_input", "chunk_id": i}
            for i in range(len(request.documents))
        ],
    )

    # Run pipeline
    result = pipeline.run(request.query)

    return RAGResponse(
        query=result["query"],
        retrieved_chunks=result["retrieved_chunks"],
        raw_response=result["raw_response"],
        hallucination_score=result["hallucination_score"],
        hallucination_detected=result["hallucination_detected"],
        final_response=result["final_response"],
        mitigation_applied=result["mitigation_applied"],
        latency_ms=result["latency_ms"],
    )


# ── Apply rate limiting if available ────────────────────────────────────

if RATE_LIMITING_ENABLED and limiter:
    detect_hallucination = limiter.limit("10/minute")(detect_hallucination)
    run_rag_pipeline = limiter.limit("10/minute")(run_rag_pipeline)


# ── Main ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    uvicorn.run(
        "app.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
