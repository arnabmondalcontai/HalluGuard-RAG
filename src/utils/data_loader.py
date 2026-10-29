"""Dataset loaders and synthetic hallucination generators.

Provides unified data loading from HuggingFace datasets (HaluEval, TruthfulQA,
Natural Questions) and a synthetic hallucination generation utility using LLM.
All loaders return a consistent schema for downstream evaluation.
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Consistent output schema for all loaders
SCHEMA_KEYS = {"query", "context", "response", "label"}

DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data"
)


def load_halueval(split: str = "data", subset: str = "qa") -> List[Dict[str, Any]]:
    """Load the HaluEval hallucination evaluation dataset.

    Args:
        split: Dataset split to load (default: "data").
        subset: HaluEval subset — "qa", "summarization", or "dialogue".

    Returns:
        List of dicts with keys: query, context, response, label.
        label=0 for faithful, label=1 for hallucinated.

    Raises:
        ImportError: If datasets library is not available.
        ValueError: If subset is not recognized.
    """
    from datasets import load_dataset

    valid_subsets = {"qa", "summarization", "dialogue"}
    if subset not in valid_subsets:
        raise ValueError(f"subset must be one of {valid_subsets}, got '{subset}'")

    logger.info("Loading HaluEval dataset (subset=%s, split=%s)...", subset, split)

    try:
        dataset = load_dataset("pminervini/HaluEval", subset, split=split)
    except Exception as e:
        logger.error("Failed to load HaluEval: %s", e)
        raise

    results = []
    for item in tqdm(dataset, desc="Processing HaluEval"):
        knowledge = item.get("knowledge", "")
        question = item.get("question", "")

        # HaluEval QA subset has 'right_answer'/'hallucinated_answer', dialogue has 'answer'/'hallucination'
        right_answer = item.get("right_answer") or item.get("answer")
        hallucinated_answer = item.get("hallucinated_answer") or item.get("hallucination")

        # Faithful sample
        if right_answer:
            results.append({
                "query": question,
                "context": [knowledge] if knowledge else [],
                "response": right_answer,
                "label": 0,
            })

        # Hallucinated sample
        if hallucinated_answer:
            results.append({
                "query": question,
                "context": [knowledge] if knowledge else [],
                "response": hallucinated_answer,
                "label": 1,
            })

    logger.info("Loaded %d samples from HaluEval.", len(results))
    return results


def load_truthfulqa(split: str = "validation") -> List[Dict[str, Any]]:
    """Load the TruthfulQA dataset for hallucination evaluation.

    Args:
        split: Dataset split (TruthfulQA only has "validation").

    Returns:
        List of dicts with keys: query, context, response, label.

    Raises:
        ImportError: If datasets library is not available.
    """
    from datasets import load_dataset

    logger.info("Loading TruthfulQA dataset (split=%s)...", split)

    try:
        dataset = load_dataset("truthfulqa/truthful_qa", "multiple_choice", split=split)
    except Exception as e:
        logger.error("Failed to load TruthfulQA: %s", e)
        raise

    results = []
    for item in tqdm(dataset, desc="Processing TruthfulQA"):
        question = item.get("question", "")

        # Best answer as faithful
        if item.get("best_answer"):
            results.append({
                "query": question,
                "context": [],  # TruthfulQA is open-domain
                "response": item["best_answer"],
                "label": 0,
            })

        # Incorrect answers as hallucinated
        incorrect_answers = item.get("incorrect_answers", [])
        for ans in incorrect_answers[:1]:  # Take first incorrect to balance
            results.append({
                "query": question,
                "context": [],
                "response": ans,
                "label": 1,
            })

    logger.info("Loaded %d samples from TruthfulQA.", len(results))
    return results


def load_natural_questions(n: int = 1000) -> List[Dict[str, Any]]:
    """Load a subset of the Natural Questions dataset.

    Args:
        n: Number of samples to load (default: 1000).

    Returns:
        List of dicts with keys: query, context, response, label.

    Raises:
        ImportError: If datasets library is not available.
    """
    from datasets import load_dataset

    logger.info("Loading Natural Questions dataset (n=%d)...", n)

    try:
        dataset = load_dataset(
            "natural_questions", "default", split=f"train[:{n}]",
            trust_remote_code=True
        )
    except Exception as e:
        logger.error("Failed to load Natural Questions: %s", e)
        raise

    results = []
    for item in tqdm(dataset, desc="Processing NQ"):
        question = item.get("question", {})
        question_text = question.get("text", "") if isinstance(question, dict) else str(question)

        # Extract document context
        document = item.get("document", {})
        doc_text = document.get("text", "") if isinstance(document, dict) else ""

        # Extract short answer
        annotations = item.get("annotations", {})
        short_answers = []
        if isinstance(annotations, dict):
            sa = annotations.get("short_answers", [])
            if sa and len(sa) > 0:
                for answer_set in sa:
                    if isinstance(answer_set, dict):
                        text = answer_set.get("text", [])
                        if text:
                            short_answers.extend(text if isinstance(text, list) else [text])

        if question_text and short_answers:
            results.append({
                "query": question_text,
                "context": [doc_text[:2000]] if doc_text else [],
                "response": short_answers[0] if short_answers else "",
                "label": 0,  # Ground truth answers are faithful
            })

    logger.info("Loaded %d samples from Natural Questions.", len(results))
    return results


def generate_synthetic_hallucinations(
    docs: List[str],
    n_per_doc: int = 3,
    llm_generator: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """Generate synthetic faithful and hallucinated QA pairs using an LLM.

    For each document, generates both faithful and hallucinated QA pairs
    and labels them accordingly. Results are saved to data/synthetic/.

    Args:
        docs: List of source document texts.
        n_per_doc: Number of QA pairs to generate per document.
        llm_generator: An LLMGenerator instance. If None, imports and creates one.

    Returns:
        List of dicts with keys: query, context, response, label.
        label=0 for faithful, label=1 for hallucinated.

    Raises:
        ValueError: If docs is empty.
        RuntimeError: If LLM generation fails for all documents.
    """
    if not docs:
        raise ValueError("docs list cannot be empty.")

    if llm_generator is None:
        from src.rag_pipeline.generator import LLMGenerator
        llm_generator = LLMGenerator()

    synthetic_data = []
    output_dir = os.path.join(DATA_DIR, "synthetic")
    os.makedirs(output_dir, exist_ok=True)

    faithful_prompt = (
        "Given the following document, generate {n} question-answer pairs where "
        "the answer is ENTIRELY supported by the document. Format as JSON array: "
        '[{{"question": "...", "answer": "..."}}]\n\nDocument:\n{doc}'
    )

    hallucinated_prompt = (
        "Given the following document, generate {n} question-answer pairs where "
        "the answer contains plausible-sounding but INCORRECT information not "
        "supported by the document. Format as JSON array: "
        '[{{"question": "...", "answer": "..."}}]\n\nDocument:\n{doc}'
    )

    for i, doc in enumerate(tqdm(docs, desc="Generating synthetic data")):
        doc_truncated = doc[:3000]  # Truncate for LLM context limits

        # Generate faithful QA pairs
        try:
            faithful_response = llm_generator.generate(
                query=faithful_prompt.format(n=n_per_doc, doc=doc_truncated),
                context_chunks=[],
                system_prompt="You are a QA pair generator. Output valid JSON only.",
            )
            faithful_pairs = _parse_qa_json(faithful_response)
            for pair in faithful_pairs[:n_per_doc]:
                synthetic_data.append({
                    "query": pair["question"],
                    "context": [doc_truncated],
                    "response": pair["answer"],
                    "label": 0,
                })
        except Exception as e:
            logger.warning("Failed to generate faithful pairs for doc %d: %s", i, e)

        # Generate hallucinated QA pairs
        try:
            hallucinated_response = llm_generator.generate(
                query=hallucinated_prompt.format(n=n_per_doc, doc=doc_truncated),
                context_chunks=[],
                system_prompt="You are a QA pair generator. Output valid JSON only.",
            )
            hallucinated_pairs = _parse_qa_json(hallucinated_response)
            for pair in hallucinated_pairs[:n_per_doc]:
                synthetic_data.append({
                    "query": pair["question"],
                    "context": [doc_truncated],
                    "response": pair["answer"],
                    "label": 1,
                })
        except Exception as e:
            logger.warning("Failed to generate hallucinated pairs for doc %d: %s", i, e)

    # Save to JSONL
    output_path = os.path.join(output_dir, "synthetic_hallucinations.jsonl")
    with open(output_path, "w", encoding="utf-8") as f:
        for item in synthetic_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    logger.info(
        "Generated %d synthetic samples. Saved to %s",
        len(synthetic_data), output_path
    )
    return synthetic_data


def _parse_qa_json(response: str) -> List[Dict[str, str]]:
    """Parse JSON array of QA pairs from LLM response.

    Args:
        response: Raw LLM response string containing JSON.

    Returns:
        List of dicts with 'question' and 'answer' keys.
    """
    # Try to extract JSON from response
    response = response.strip()

    # Handle markdown code blocks
    if "```json" in response:
        response = response.split("```json")[1].split("```")[0].strip()
    elif "```" in response:
        response = response.split("```")[1].split("```")[0].strip()

    # Find JSON array boundaries
    start_idx = response.find("[")
    end_idx = response.rfind("]")
    if start_idx != -1 and end_idx != -1:
        response = response[start_idx:end_idx + 1]

    try:
        pairs = json.loads(response)
        if isinstance(pairs, list):
            return [p for p in pairs if "question" in p and "answer" in p]
    except json.JSONDecodeError:
        logger.warning("Failed to parse QA JSON from LLM response.")

    return []
