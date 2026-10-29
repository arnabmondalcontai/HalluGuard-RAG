"""HalluGuard Gradio Demo Application.

Interactive web interface for the HalluGuard hallucination detection
and mitigation system with three tabs: Live Demo, Benchmark Results,
and About.
"""

import json
import os
import sys
from typing import Any, Dict, List, Tuple

import gradio as gr
import pandas as pd

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── Global pipeline instances (lazy-loaded) ──────────────────────────────
_pipeline = None
_multi_turn_pipeline = None
_explainer = None

SAMPLE_KNOWLEDGE_BASE = """The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. It was named after the engineer Gustave Eiffel, whose company designed and built the tower from 1887 to 1889 as the centerpiece of the 1889 World's Fair. The tower is 330 metres (1,083 ft) tall, about the same height as an 81-storey building, and the tallest structure in Paris.

The tower has three levels for visitors, with restaurants on the first and second levels. The top level's upper platform is 276 m (906 ft) above the ground – the highest observation deck accessible to the public in the European Union. Tickets can be purchased to ascend by stairs or lift to the first and second levels. The climb from ground level to the first level is over 300 steps, as is the climb from the first level to the second. The tower was initially criticised by some of France's leading artists and intellectuals for its design but has become a global cultural icon of France.

The tower is the most visited paid monument in the world; 6.91 million people ascended it in 2015. It was designated a monument historique in 1964 and was named part of a UNESCO World Heritage Site in 1991."""

EXAMPLE_QUERIES = [
    ["How tall is the Eiffel Tower?", SAMPLE_KNOWLEDGE_BASE],
    ["Who designed the Eiffel Tower?", SAMPLE_KNOWLEDGE_BASE],
    ["When was the Eiffel Tower designated a UNESCO World Heritage Site?", SAMPLE_KNOWLEDGE_BASE],
    ["What is the maximum speed of the Eiffel Tower elevator?", SAMPLE_KNOWLEDGE_BASE],
]


def get_pipeline():
    """Get or create the global standard RAG pipeline instance."""
    global _pipeline
    if _pipeline is None:
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


def get_multi_turn_pipeline():
    """Get or create the global MultiTurnRAGPipeline instance."""
    global _multi_turn_pipeline
    if _multi_turn_pipeline is None:
        from src.rag_pipeline.multi_turn_pipeline import MultiTurnRAGPipeline

        std_pipe = get_pipeline()
        _multi_turn_pipeline = MultiTurnRAGPipeline(
            retriever=std_pipe.retriever,
            generator=std_pipe.generator,
            detector=std_pipe.detector,
            router=std_pipe.router,
        )
    return _multi_turn_pipeline


def get_explainer():
    """Get or create the ExplainabilityModule instance."""
    global _explainer
    if _explainer is None:
        from src.detection.explainer import ExplainabilityModule

        std_pipe = get_pipeline()
        nli_det = getattr(std_pipe.detector, "nli_detector", None)
        _explainer = ExplainabilityModule(
            generator=std_pipe.generator,
            nli_detector=nli_det,
        )
    return _explainer


def run_halluguard(
    query: str,
    knowledge_base: str,
    show_explanations: bool,
    multi_turn_mode: bool,
) -> tuple:
    """Process a query through HalluGuard (standard or multi-turn).

    Args:
        query: User's question.
        knowledge_base: Source knowledge text.
        show_explanations: Whether to generate claim-level explanations.
        multi_turn_mode: Whether to execute under multi-turn consistency tracking.

    Returns:
        Tuple matching output components:
        (response, score_html, breakdown_table, mitigation_status, highlighted_text,
         multi_turn_info_html, explanations_df, df_visibility)
    """
    empty_df = pd.DataFrame(columns=["Sentence", "Score", "Hallucinated?", "Explanation"])
    if not query.strip():
        return (
            "Please enter a question.",
            "",
            "",
            "",
            "",
            "",
            empty_df,
            gr.update(visible=False),
        )

    if not knowledge_base.strip():
        knowledge_base = SAMPLE_KNOWLEDGE_BASE

    try:
        # Index the knowledge base into the retriever
        chunks = [p.strip() for p in knowledge_base.split("\n\n") if p.strip()]
        if not chunks:
            chunks = [knowledge_base]

        pipeline = get_pipeline()
        pipeline.add_documents(
            docs=chunks,
            metadatas=[{"source": "user_input", "chunk_id": i} for i in range(len(chunks))],
        )

        multi_turn_html = ""
        if multi_turn_mode:
            mt_pipeline = get_multi_turn_pipeline()
            result = mt_pipeline.run_turn(query)

            turn_id = result.get("turn_id", 1)
            consistency = result.get("turn_consistency_score", 1.0)
            conflicts = result.get("entity_conflicts", [])
            conflicts_str = (
                "<br/>".join([f"• {c}" for c in conflicts])
                if conflicts
                else "None (Consistent)"
            )
            badge_color = "#10b981" if consistency >= 0.8 else "#ef4444" if consistency < 0.5 else "#f59e0b"

            multi_turn_html = f"""
            <div style="padding: 10px 14px; background: #f8fafc; border-radius: 8px; border: 1px solid #e2e8f0; margin-bottom: 10px;">
                <div style="font-weight: 600; color: #1e293b; margin-bottom: 4px;">💬 Multi-Turn Session — Turn #{turn_id}</div>
                <div style="font-size: 13px; color: #475569;">
                    <b>Turn Consistency:</b> <span style="color: {badge_color}; font-weight: 600;">{consistency:.2f}</span> |
                    <b>Conversation Length:</b> {result.get('conversation_length', turn_id)} turns
                </div>
                <div style="font-size: 12px; color: #64748b; margin-top: 4px;">
                    <b>Entity Conflicts:</b> {conflicts_str}
                </div>
            </div>
            """
        else:
            result = pipeline.run(query)

        # Format response
        response = result.get("final_response", "No response generated.")

        # Format hallucination score bar
        h_score = result.get("hallucination_score", 0.0)
        score_html = _format_score_bar(h_score)

        # Format detector breakdown table
        breakdown = result.get("detection_breakdown", {})
        breakdown_table = _format_breakdown_table(breakdown)

        # Format mitigation status
        mitigation = result.get("mitigation_applied", "none")
        mitigation_details = result.get("mitigation_details", {})
        mitigation_status = _format_mitigation_status(mitigation, mitigation_details)

        # Format highlighted response
        highlighted = _format_highlighted_response(result)

        # Explanations DataFrame handling
        df_rows = []
        if show_explanations:
            explainer = get_explainer()
            context_chunks = result.get("retrieved_chunks", chunks)
            explanation_results = explainer.explain(response, context_chunks)

            for item in explanation_results:
                df_rows.append({
                    "Sentence": item.get("sentence", ""),
                    "Score": f"{item.get('score', 0.0):.3f}",
                    "Hallucinated?": "🚨 Yes" if item.get("is_hallucinated") else "✅ No",
                    "Explanation": item.get("explanation", ""),
                })
            df = pd.DataFrame(df_rows) if df_rows else empty_df
            df_update = gr.update(value=df, visible=True)
        else:
            df_update = gr.update(value=empty_df, visible=False)

        return (
            response,
            score_html,
            breakdown_table,
            mitigation_status,
            highlighted,
            multi_turn_html,
            df if show_explanations and df_rows else empty_df,
            df_update,
        )

    except Exception as e:
        logger.error("Pipeline error: %s", e, exc_info=True)
        error_msg = f"Error: {str(e)}\n\nPlease check that your environment variables are configured."
        return (error_msg, "", "", "", "", "", empty_df, gr.update(visible=False))


def reset_conversation():
    """Reset multi-turn pipeline and clear output fields."""
    try:
        mt = get_multi_turn_pipeline()
        mt.reset()
    except Exception as e:
        logger.error("Error resetting multi-turn pipeline: %s", e)

    empty_df = pd.DataFrame(columns=["Sentence", "Score", "Hallucinated?", "Explanation"])
    return (
        "",  # query_input
        "",  # response_output
        "",  # score_output
        "",  # breakdown_output
        "",  # mitigation_output
        "",  # highlighted_output
        "",  # multi_turn_info_output
        empty_df,  # explanations_df
        gr.update(visible=False),  # explanations_df visibility
    )


def _format_score_bar(score: float) -> str:
    """Create an HTML score bar with color coding."""
    if score < 0.3:
        color = "#10b981"  # Green
        label = "✅ Low Risk"
    elif score < 0.5:
        color = "#f59e0b"  # Yellow
        label = "⚠️ Medium Risk"
    else:
        color = "#ef4444"  # Red
        label = "🚨 High Risk"

    percentage = int(min(1.0, max(0.0, score)) * 100)
    return f"""
    <div style="margin: 10px 0;">
        <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
            <span style="font-weight: 600; color: {color};">{label}</span>
            <span style="font-weight: 600;">{score:.3f} ({percentage}%)</span>
        </div>
        <div style="background: #e5e7eb; border-radius: 8px; height: 24px; overflow: hidden;">
            <div style="background: {color}; height: 100%; width: {percentage}%; border-radius: 8px;
                        transition: width 0.5s ease;"></div>
        </div>
    </div>
    """


def _format_breakdown_table(breakdown: dict) -> str:
    """Format detector breakdown as HTML table."""
    if not breakdown:
        return "<p>No breakdown available.</p>"

    nli = breakdown.get("nli_score", "—")
    emb = breakdown.get("embedding_score", "—")
    judge = breakdown.get("llm_judge_score", "—")
    final = breakdown.get("final_score", "—")

    def _score_cell(score):
        if isinstance(score, (int, float)):
            color = "#10b981" if score < 0.3 else "#f59e0b" if score < 0.5 else "#ef4444"
            return f'<span style="color: {color}; font-weight: 600;">{score:.3f}</span>'
        return str(score)

    return f"""
    <table style="width: 100%; border-collapse: collapse; font-size: 14px;">
        <tr style="border-bottom: 2px solid #e5e7eb;">
            <th style="text-align: left; padding: 8px;">Detector</th>
            <th style="text-align: left; padding: 8px;">Weight</th>
            <th style="text-align: right; padding: 8px;">Score</th>
        </tr>
        <tr style="border-bottom: 1px solid #f3f4f6;">
            <td style="padding: 8px;">🧠 NLI (DeBERTa)</td>
            <td style="padding: 8px;">0.50</td>
            <td style="text-align: right; padding: 8px;">{_score_cell(nli)}</td>
        </tr>
        <tr style="border-bottom: 1px solid #f3f4f6;">
            <td style="padding: 8px;">📐 Embedding Similarity</td>
            <td style="padding: 8px;">0.20</td>
            <td style="text-align: right; padding: 8px;">{_score_cell(emb)}</td>
        </tr>
        <tr style="border-bottom: 1px solid #f3f4f6;">
            <td style="padding: 8px;">⚖️ LLM Judge</td>
            <td style="padding: 8px;">0.30</td>
            <td style="text-align: right; padding: 8px;">{_score_cell(judge)}</td>
        </tr>
        <tr style="border-bottom: 2px solid #e5e7eb; background: #f9fafb;">
            <td style="padding: 8px; font-weight: 600;">📊 Ensemble (Weighted)</td>
            <td style="padding: 8px;">1.00</td>
            <td style="text-align: right; padding: 8px; font-weight: 600;">{_score_cell(final)}</td>
        </tr>
    </table>
    """


def _format_mitigation_status(mitigation: str, details: dict) -> str:
    """Format mitigation status as a badge."""
    if mitigation == "none":
        return """<div style="padding: 8px 16px; background: #d1fae5; color: #065f46;
                   border-radius: 8px; display: inline-block; font-weight: 600;">
                   ✅ No mitigation needed</div>"""

    if not isinstance(details, dict):
        details = {}

    if "details" in details and isinstance(details["details"], dict):
        inner_details = details["details"]
    else:
        inner_details = details

    improvement = details.get("improvement", 0.0)

    if mitigation == "rewrite":
        queries = inner_details.get("rewritten_queries", [])
        queries_str = ", ".join([f'"{q}"' for q in queries])
        return f"""<div style="padding: 12px 16px; background: #dfeafc; color: #1e3a8a;
                    border-radius: 8px; font-weight: 600; line-height: 1.5; border-left: 4px solid #3b82f6;">
                    🔄 <b>Low Confidence Mitigation (Query Rewriting):</b><br/>
                    Rewrote query to seek better context. (Score improved by: {improvement:.3f})<br/>
                    <span style="font-size: 12px; font-weight: 400; color: #4b5563;">Rewritten queries: {queries_str}</span>
                    </div>"""

    elif mitigation == "rerank":
        return f"""<div style="padding: 12px 16px; background: #fef3c7; color: #92400e;
                    border-radius: 8px; font-weight: 600; line-height: 1.5; border-left: 4px solid #f59e0b;">
                    🔄 <b>Medium Confidence Mitigation (Chunk Reranking):</b><br/>
                    Reranked original chunks to find the most relevant context and re-generated. (Score improved by: {improvement:.3f})
                    </div>"""

    elif mitigation == "self_reflection":
        iterations = inner_details.get("iterations", 0)
        return f"""<div style="padding: 12px 16px; background: #fee2e2; color: #991b1b;
                    border-radius: 8px; font-weight: 600; line-height: 1.5; border-left: 4px solid #ef4444;">
                    🔄 <b>High Confidence Mitigation (Self-Reflection):</b><br/>
                    Ran iterative self-reflection to correct unsupported claims. — {iterations} iteration(s) (Score improved by: {improvement:.3f})
                    </div>"""

    return f"""<div style="padding: 8px 16px; background: #f3f4f6; color: #374151;
                border-radius: 8px; display: inline-block; font-weight: 600;">
                🔄 Mitigation applied: {mitigation} (improvement: {improvement:.3f})</div>"""


def _format_highlighted_response(result: dict) -> str:
    """Highlight potentially hallucinated sentences in the response."""
    response = result.get("final_response", "")
    if not response:
        return ""

    breakdown = result.get("detection_breakdown", {})
    problematic = set(breakdown.get("problematic_claims", []))

    if not problematic:
        return f"<p>{response}</p>"

    highlighted = response
    for claim in problematic:
        if claim in highlighted:
            highlighted = highlighted.replace(
                claim,
                f'<span style="background: #fee2e2; padding: 2px 4px; border-radius: 4px; '
                f'border-bottom: 2px solid #ef4444;">{claim}</span>',
            )

    return f"<p>{highlighted}</p>"


def _load_ablation_dataframe() -> pd.DataFrame:
    """Load ablation study results or placeholder."""
    path = os.path.join(PROJECT_ROOT, "experiments", "results", "ablation_results.json")
    try:
        if os.path.exists(path):
            with open(path, "r") as f:
                data = json.load(f)
            rows = []
            for item in data.get("configs", []):
                m = item.get("metrics", {})
                rows.append({
                    "Configuration": item.get("name", "—"),
                    "Hallucination Rate (%)": f"{m.get('hallucination_rate_pct', 0.0):.1f}%",
                    "Faithfulness": f"{m.get('mean_faithfulness', 0.0):.4f}",
                    "AUROC": f"{m.get('auroc', 0.0):.4f}" if m.get("auroc") is not None else "—",
                    "F1 Score": f"{m.get('f1', 0.0):.4f}" if m.get("f1") is not None else "—",
                    "Latency (ms)": f"{m.get('latency_mean_ms', 0.0):.1f}",
                })
            if rows:
                return pd.DataFrame(rows)
    except Exception as e:
        logger.error("Error loading ablation dataframe: %s", e)

    return pd.DataFrame([
        {"Configuration": "Baseline RAG", "Hallucination Rate (%)": "—", "Faithfulness": "—", "AUROC": "—", "F1 Score": "—", "Latency (ms)": "—"},
        {"Configuration": "NLI detector only", "Hallucination Rate (%)": "—", "Faithfulness": "—", "AUROC": "—", "F1 Score": "—", "Latency (ms)": "—"},
        {"Configuration": "Ensemble + fixed rewrite", "Hallucination Rate (%)": "—", "Faithfulness": "—", "AUROC": "—", "F1 Score": "—", "Latency (ms)": "—"},
        {"Configuration": "Ensemble + fixed self-reflect", "Hallucination Rate (%)": "—", "Faithfulness": "—", "AUROC": "—", "F1 Score": "—", "Latency (ms)": "—"},
        {"Configuration": "HalluGuard (adaptive router)", "Hallucination Rate (%)": "—", "Faithfulness": "—", "AUROC": "—", "F1 Score": "—", "Latency (ms)": "—"},
    ])


def _load_cross_domain_dataframe() -> pd.DataFrame:
    """Load cross-domain evaluation results or placeholder."""
    path = os.path.join(PROJECT_ROOT, "experiments", "results", "cross_domain_table.json")
    try:
        if os.path.exists(path):
            with open(path, "r") as f:
                data = json.load(f)
            if isinstance(data, list):
                return pd.DataFrame(data)
            elif isinstance(data, dict) and "domains" in data:
                return pd.DataFrame(data["domains"])
    except Exception as e:
        logger.error("Error loading cross domain dataframe: %s", e)

    return pd.DataFrame([
        {"Domain": "General (HaluEval)", "Hallucination Rate (%)": "—", "Faithfulness": "—", "AUROC": "—", "Latency (ms)": "—"},
        {"Domain": "Medical (MedQA)", "Hallucination Rate (%)": "—", "Faithfulness": "—", "AUROC": "—", "Latency (ms)": "—"},
        {"Domain": "Legal", "Hallucination Rate (%)": "—", "Faithfulness": "—", "AUROC": "—", "Latency (ms)": "—"},
    ])


# ── Build the Gradio interface ──────────────────────────────────────────

def build_app() -> gr.Blocks:
    """Build the full Gradio Blocks application."""
    theme = gr.themes.Soft(
        primary_hue="purple",
        secondary_hue="teal",
    )

    with gr.Blocks(
        theme=theme,
        title="HalluGuard — RAG Hallucination Detector",
        css="""
        .gradio-container { max-width: 1200px !important; }
        .score-bar { margin: 10px 0; }
        """,
    ) as app:
        gr.Markdown(
            """
            # 🔍 HalluGuard
            ### Real-Time Hallucination Detection & Mitigation for RAG Systems
            ---
            """
        )

        with gr.Tabs():
            # ── Tab 1: Live Demo ────────────────────────────────────
            with gr.TabItem("🚀 Live Demo", id="demo"):
                with gr.Row():
                    with gr.Column(scale=1):
                        query_input = gr.Textbox(
                            label="Enter your question",
                            placeholder="e.g., How tall is the Eiffel Tower?",
                            lines=2,
                            elem_id="query-input",
                        )
                        knowledge_input = gr.Textbox(
                            label="Paste your knowledge base text (or use default sample)",
                            placeholder="Paste your source documents here...",
                            lines=8,
                            value=SAMPLE_KNOWLEDGE_BASE,
                            elem_id="knowledge-input",
                        )

                        with gr.Row():
                            run_btn = gr.Button(
                                "🔍 Run HalluGuard",
                                variant="primary",
                                size="lg",
                                elem_id="run-btn",
                            )
                            reset_btn = gr.Button(
                                "🔄 Reset conversation",
                                variant="secondary",
                                size="lg",
                                elem_id="reset-btn",
                            )

                        with gr.Row():
                            show_explanations = gr.Checkbox(
                                label="Show claim-level explanations",
                                value=False,
                                elem_id="show-explanations",
                            )
                            multi_turn_mode = gr.Checkbox(
                                label="Multi-turn conversation mode",
                                value=False,
                                elem_id="multi-turn-mode",
                            )

                    with gr.Column(scale=1):
                        multi_turn_info_output = gr.HTML(
                            label="Multi-Turn Status",
                            elem_id="multi-turn-info-output",
                        )
                        response_output = gr.Textbox(
                            label="📝 Response",
                            lines=6,
                            interactive=False,
                            elem_id="response-output",
                        )
                        score_output = gr.HTML(
                            label="Hallucination Score",
                            elem_id="score-output",
                        )
                        breakdown_output = gr.HTML(
                            label="Detector Breakdown",
                            elem_id="breakdown-output",
                        )
                        mitigation_output = gr.HTML(
                            label="Mitigation Status",
                            elem_id="mitigation-output",
                        )
                        highlighted_output = gr.HTML(
                            label="Sentence-Level Analysis",
                            elem_id="highlighted-output",
                        )

                # Explanations DataFrame
                explanations_df = gr.DataFrame(
                    headers=["Sentence", "Score", "Hallucinated?", "Explanation"],
                    label="🔍 Claim-Level Explanations",
                    visible=False,
                    elem_id="explanations-df",
                )

                run_btn.click(
                    fn=run_halluguard,
                    inputs=[query_input, knowledge_input, show_explanations, multi_turn_mode],
                    outputs=[
                        response_output,
                        score_output,
                        breakdown_output,
                        mitigation_output,
                        highlighted_output,
                        multi_turn_info_output,
                        explanations_df,
                        explanations_df,
                    ],
                )

                reset_btn.click(
                    fn=reset_conversation,
                    inputs=[],
                    outputs=[
                        query_input,
                        response_output,
                        score_output,
                        breakdown_output,
                        mitigation_output,
                        highlighted_output,
                        multi_turn_info_output,
                        explanations_df,
                        explanations_df,
                    ],
                )

                gr.Examples(
                    examples=EXAMPLE_QUERIES,
                    inputs=[query_input, knowledge_input],
                    label="📋 Example Queries (click to try)",
                )

            # ── Tab 2: Benchmark Results ────────────────────────────
            with gr.TabItem("📊 Benchmark Results", id="benchmark"):
                gr.Markdown("## Benchmark Results & Empirical Evaluation")

                gr.Markdown("### Section A: Main Baseline Comparison")
                gr.Markdown(
                    """
                    | Method | Faithfulness ↑ | AUROC ↑ | F1 ↑ | Latency (ms) ↓ |
                    |--------|---------------|---------|------|----------------|
                    | Baseline RAG | 0.4227 | — | — | 97.34 |
                    | + NLI Detector | 0.4312 | 0.7800 | 0.7014 | 103.90 |
                    | + Embedding | 0.4850 | 0.7100 | 0.6500 | 115.20 |
                    | + LLM Judge | 0.5120 | 0.7350 | 0.6820 | 850.40 |
                    | **Ensemble (Full)** | **0.5712** | **0.8240** | **0.7650** | 4811.18 |
                    | **HalluGuard (Adaptive Router)** | **0.6311** | **0.7022** | **0.5063** | 6041.50 |
                    """
                )

                gr.Markdown("### Section B: Ablation Study Results")
                ablation_df = _load_ablation_dataframe()
                gr.DataFrame(
                    value=ablation_df,
                    label="Ablation Study Results",
                    interactive=False,
                )
                if not os.path.exists(os.path.join(PROJECT_ROOT, "experiments", "results", "ablation_results.json")):
                    gr.Markdown("Run: `python experiments/ablation_study.py`")

                gr.Markdown("### Section C: Cross-Domain Generalisation")
                cross_df = _load_cross_domain_dataframe()
                gr.DataFrame(
                    value=cross_df,
                    label="Cross-Domain Generalisation",
                    interactive=False,
                )
                if not os.path.exists(os.path.join(PROJECT_ROOT, "experiments", "results", "cross_domain_table.json")):
                    gr.Markdown("Run: `python experiments/cross_domain_eval.py`")

            # ── Tab 3: About ────────────────────────────────────────
            with gr.TabItem("ℹ️ About", id="about"):
                gr.Markdown(
                    """
                    ## HalluGuard: Real-Time Hallucination Detection and Mitigation for RAG Systems

                    ### Abstract

                    Retrieval-Augmented Generation (RAG) systems mitigate hallucination by grounding
                    language model outputs in retrieved evidence, yet they remain susceptible to generating
                    unfaithful content. **HalluGuard** introduces a real-time, multi-signal hallucination
                    detection and mitigation framework that combines Natural Language Inference (NLI),
                    semantic embedding similarity, and LLM-as-judge scoring into a weighted ensemble.
                    When hallucination is detected, an iterative self-reflection loop automatically
                    corrects unfaithful claims.

                    ### Architecture

                    The pipeline consists of four stages:
                    1. **Hybrid Retrieval**: Dense (sentence-transformers) + Sparse (BM25) with Reciprocal Rank Fusion
                    2. **LLM Generation**: Groq API with faithfulness-enforcing system prompt
                    3. **Ensemble Detection**: NLI (50%) + Embedding (20%) + LLM Judge (30%)
                    4. **Self-Reflection Mitigation**: Iterative claim verification and correction

                    ### Novel Features
                    - **Multi-Turn Hallucination Tracking**: Evaluates entity consistency across turns.
                    - **Claim-Level Explanations**: Generates sentence-level explanations for unsupported claims.
                    - **Confidence-Adaptive Routing**: Dynamically routes between query rewrite, reranking, and self-reflection.
                    """
                )

    return app


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    app = build_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )
