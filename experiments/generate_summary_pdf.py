#!/usr/bin/env python3
"""Generate a professional PDF summary of the HalluGuard project."""

import os
import sys
from fpdf import FPDF

OUTPUT_PATH = "/Users/arnabmondal/Desktop/Rag conntrol/HalluGuard_Project_Summary.pdf"


class HalluGuardPDF(FPDF):
    """Custom PDF with header/footer styling."""

    def header(self):
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(100, 100, 100)
        self.cell(0, 8, "HalluGuard - Project Summary", align="L")
        self.ln(4)
        self.set_draw_color(39, 174, 96)
        self.set_line_width(0.8)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(6)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    def section_title(self, title):
        self.ln(4)
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(44, 62, 80)
        self.cell(0, 10, title, new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(39, 174, 96)
        self.set_line_width(0.4)
        self.line(10, self.get_y(), 80, self.get_y())
        self.ln(3)

    def sub_title(self, title):
        self.ln(2)
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(52, 73, 94)
        self.cell(0, 8, title, new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def body_text(self, text):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(60, 60, 60)
        self.multi_cell(0, 5.5, text)
        self.ln(1)

    def bullet(self, text):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(60, 60, 60)
        x_start = self.get_x()
        self.cell(6, 5.5, "-")
        x_after = self.get_x()
        self.multi_cell(190 - (x_after - 10), 5.5, text)
        self.set_x(x_start)

    def add_table(self, headers, rows, col_widths=None):
        if col_widths is None:
            col_widths = [190 / len(headers)] * len(headers)

        # Header row
        self.set_font("Helvetica", "B", 9)
        self.set_fill_color(44, 62, 80)
        self.set_text_color(255, 255, 255)
        for i, h in enumerate(headers):
            self.cell(col_widths[i], 7, h, border=1, fill=True, align="C")
        self.ln()

        # Data rows
        self.set_font("Helvetica", "", 9)
        self.set_text_color(60, 60, 60)
        fill = False
        for row in rows:
            if fill:
                self.set_fill_color(240, 245, 250)
            else:
                self.set_fill_color(255, 255, 255)
            for i, cell in enumerate(row):
                align = "L" if i == 0 else "C"
                self.cell(col_widths[i], 6.5, str(cell), border=1, fill=True, align=align)
            self.ln()
            fill = not fill
        self.ln(2)


def build_pdf():
    pdf = HalluGuardPDF()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=20)

    # ── Page 1: Title page ───────────────────────────────────────────
    pdf.add_page()
    pdf.ln(35)
    pdf.set_font("Helvetica", "B", 28)
    pdf.set_text_color(44, 62, 80)
    pdf.cell(0, 15, "HalluGuard", align="C", new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("Helvetica", "", 14)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 10, "Real-Time Hallucination Detection & Mitigation", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 8, "for RAG Systems", align="C", new_x="LMARGIN", new_y="NEXT")

    pdf.ln(10)
    pdf.set_draw_color(39, 174, 96)
    pdf.set_line_width(1.2)
    pdf.line(60, pdf.get_y(), 150, pdf.get_y())

    pdf.ln(15)
    pdf.set_font("Helvetica", "", 12)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 8, "Author: Arnab Mondal", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 8, "License: MIT", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 8, "Python 3.13 | PyTorch | Groq | HuggingFace", align="C", new_x="LMARGIN", new_y="NEXT")

    # ── Page 2: Abstract & Architecture ──────────────────────────────
    pdf.add_page()

    pdf.section_title("1. Abstract")
    pdf.body_text(
        "Retrieval-Augmented Generation (RAG) systems mitigate hallucination by grounding "
        "language model outputs in retrieved evidence, yet they remain susceptible to generating "
        "unfaithful content - especially when retrieved context is noisy or incomplete.\n\n"
        "HalluGuard introduces a real-time, multi-signal hallucination detection and mitigation "
        "framework that combines Natural Language Inference (NLI), semantic embedding similarity, "
        "and LLM-as-judge scoring into a weighted ensemble. When hallucination is detected, a "
        "confidence-adaptive router selects the appropriate mitigation strategy - from query "
        "rewriting for borderline cases to iterative self-reflection for severe hallucinations."
    )

    pdf.section_title("2. Architecture Overview")
    pdf.body_text(
        "The pipeline processes each query through the following stages:"
    )
    pdf.bullet("User Query -> Hybrid Retriever (Dense embeddings + BM25 + Reciprocal Rank Fusion)")
    pdf.bullet("Retrieved context + Query -> LLM Generator (Groq / LLaMA 3.1 8B Instant)")
    pdf.bullet("Generated response -> Ensemble Detector (NLI + Embedding + LLM Judge in parallel)")
    pdf.bullet("If hallucination detected -> Confidence-Adaptive Mitigation Router")
    pdf.ln(3)

    pdf.sub_title("Ensemble Detector Weights")
    pdf.add_table(
        ["Detector", "Weight", "Type", "Model"],
        [
            ["NLI (Entailment)", "0.50", "Local (CPU)", "DeBERTa v3 cross-encoder"],
            ["Embedding Similarity", "0.20", "Local (CPU)", "all-mpnet-base-v2"],
            ["LLM Judge", "0.30", "API (Groq)", "LLaMA 3.1 8B Instant"],
        ],
        col_widths=[45, 18, 35, 92],
    )

    pdf.sub_title("Mitigation Router Tiers")
    pdf.add_table(
        ["Score Range", "Strategy", "Description"],
        [
            ["< 0.50", "None", "Response is faithful - return as-is"],
            ["0.50 - 0.65", "Query Rewriting", "Rephrase query for better retrieval"],
            ["0.65 - 0.80", "Context Reranking", "Reorder context chunks by relevance"],
            ["> 0.80", "Self-Reflection", "Iterative LLM self-correction (max 3 iters)"],
        ],
        col_widths=[30, 40, 120],
    )

    # ── Page 3: Project Structure ────────────────────────────────────
    pdf.add_page()

    pdf.section_title("3. Project Structure")
    pdf.add_table(
        ["Module", "Key Files", "Purpose"],
        [
            ["src/detection/", "nli_detector, embedding_detector, llm_judge, ensemble", "3-signal hallucination detection"],
            ["src/mitigation/", "router, query_rewriter, reranker, self_reflection", "Tiered mitigation strategies"],
            ["src/rag_pipeline/", "retriever, generator, pipeline", "Hybrid retrieval + LLM generation"],
            ["src/evaluation/", "benchmark, metrics", "Benchmarking & metrics (AUROC, F1)"],
            ["app/", "app.py, api.py", "Gradio demo UI + FastAPI REST API"],
            ["experiments/", "quick_demo, plot_results, print_metrics_table", "Evaluation scripts & charts"],
        ],
        col_widths=[35, 80, 75],
    )

    # ── Technologies ─────────────────────────────────────────────────
    pdf.section_title("4. Technologies Used")

    pdf.sub_title("Core AI / ML Stack")
    pdf.add_table(
        ["Technology", "Version", "Purpose"],
        [
            ["Python", "3.13", "Core language"],
            ["PyTorch", ">= 2.3", "Deep learning backend for transformer models"],
            ["HuggingFace Transformers", "5.13.0", "NLI model (DeBERTa v3 cross-encoder)"],
            ["Sentence-Transformers", ">= 3.0", "Embedding model (all-mpnet-base-v2)"],
            ["Groq API", ">= 0.9", "Fast LLM inference (LLaMA 3.1 8B Instant)"],
            ["scikit-learn", ">= 1.5", "AUROC, classification metrics"],
        ],
        col_widths=[55, 25, 110],
    )

    pdf.sub_title("RAG Infrastructure")
    pdf.add_table(
        ["Technology", "Purpose"],
        [
            ["LangChain", "RAG orchestration primitives"],
            ["ChromaDB", "Dense vector storage and retrieval"],
            ["FAISS (CPU)", "Alternative vector similarity search"],
            ["rank_bm25", "BM25 sparse keyword retrieval"],
            ["NLTK", "Sentence tokenization for NLI"],
        ],
        col_widths=[55, 135],
    )

    pdf.sub_title("Application & Deployment")
    pdf.add_table(
        ["Technology", "Purpose"],
        [
            ["Gradio", "Interactive web demo UI"],
            ["FastAPI + Uvicorn", "Production REST API endpoint"],
            ["SlowAPI", "API rate limiting"],
            ["Docker", "Containerized deployment"],
            ["HuggingFace Spaces", "Cloud hosting platform"],
        ],
        col_widths=[55, 135],
    )

    # ── Page 4: Evaluation & Benchmarks ──────────────────────────────
    pdf.add_page()

    pdf.sub_title("Evaluation & Data")
    pdf.add_table(
        ["Technology", "Purpose"],
        [
            ["RAGAS", "RAG evaluation metrics framework"],
            ["HaluEval (HuggingFace)", "Hallucination benchmark dataset (QA split)"],
            ["Weights & Biases (wandb)", "Experiment tracking and visualization"],
            ["Matplotlib", "Result charts and visualizations"],
            ["NumPy / Pandas / SciPy", "Numerical computation and data handling"],
        ],
        col_widths=[55, 135],
    )

    pdf.sub_title("Performance Optimizations")
    pdf.add_table(
        ["Optimization", "Impact"],
        [
            ["ThreadPoolExecutor (3 workers)", "Parallel detection: time = max(NLI, Emb, Judge)"],
            ["Singleton model caching", "Models load once, ~11s cold start eliminated"],
            ["Warmup before timed loop", "Accurate per-sample latency measurement"],
        ],
        col_widths=[60, 130],
    )

    # ── Benchmark Results ────────────────────────────────────────────
    pdf.section_title("5. Benchmark Results (HaluEval, 50 samples)")
    pdf.add_table(
        ["Metric", "Baseline RAG", "HalluGuard", "Improvement"],
        [
            ["Hallucination Rate", "64.00%", "24.00%", "-40 points (62.5% relative)"],
            ["Mean Faithfulness", "0.3571", "0.6287", "+0.2716"],
            ["AUROC", "0.8096", "0.6784", "-"],
            ["Mean Latency", "283 ms", "~4000 ms*", "Detection + mitigation cost"],
        ],
        col_widths=[40, 35, 35, 80],
    )
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(120, 120, 120)
    pdf.cell(0, 5, "* HalluGuard latency includes ensemble detection + mitigation. Baseline is NLI-only scoring.", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    pdf.sub_title("Results from Previous Run (pre-optimization)")
    pdf.add_table(
        ["Metric", "Value"],
        [
            ["Total wall time", "1005.34 seconds (~16.8 min)"],
            ["Mean latency (Baseline)", "283.65 ms"],
            ["Mean latency (HalluGuard)", "20,108 ms (~20s per sample)"],
            ["p95 latency (HalluGuard)", "44,139 ms"],
        ],
        col_widths=[55, 135],
    )

    # ── Key Features ─────────────────────────────────────────────────
    pdf.section_title("6. Key Features")
    pdf.bullet("3-Signal Ensemble Detection: NLI entailment + embedding cosine similarity + LLM-as-judge faithfulness scoring")
    pdf.bullet("Parallel Execution: All 3 detectors run simultaneously via ThreadPoolExecutor with 30s timeout per detector")
    pdf.bullet("Confidence-Adaptive Router: Automatically selects mitigation strategy based on hallucination severity")
    pdf.bullet("Iterative Self-Reflection: LLM identifies and corrects its own unsupported claims (up to 3 iterations)")
    pdf.bullet("Singleton Model Caching: NLI (DeBERTa) and Embedding (mpnet) models load once and persist across calls")
    pdf.bullet("Production-Ready: Gradio demo, FastAPI endpoint, Docker container, HuggingFace Spaces deployment")
    pdf.bullet("Comprehensive Evaluation: HaluEval benchmark with AUROC, faithfulness, hallucination rate, and latency metrics")

    # ── Save ─────────────────────────────────────────────────────────
    pdf.output(OUTPUT_PATH)
    print(f"PDF saved to: {OUTPUT_PATH}")
    print(f"File size: {os.path.getsize(OUTPUT_PATH):,} bytes")


if __name__ == "__main__":
    build_pdf()
