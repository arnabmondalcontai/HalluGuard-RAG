#!/usr/bin/env python3
"""Quick timing test for parallel ensemble detection."""

import time
import os
import sys

# Ensure project root is on sys.path
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv()

from src.detection.ensemble import EnsembleDetector

detector = EnsembleDetector()

query = "When did Einstein win the Nobel Prize?"
response = "Einstein won the Nobel Prize in 1922."
context = ["Einstein received the Nobel Prize in Physics in 1921 for his discovery of the photoelectric effect."]

print("Running parallel ensemble detection...")
start = time.time()
result = detector.score(query, response, context)
elapsed = time.time() - start

print(f"Time taken: {elapsed:.2f} seconds")
print(f"NLI score: {result['nli_score']}")
print(f"Embedding score: {result['embedding_score']}")
print(f"LLM judge score: {result['llm_judge_score']}")
print(f"Final score: {result['final_score']}")
print(f"Is hallucinated: {result['is_hallucinated']}")
print("SUCCESS - parallel detection working")
