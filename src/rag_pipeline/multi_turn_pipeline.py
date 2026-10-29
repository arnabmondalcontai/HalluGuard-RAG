"""Multi-turn RAG pipeline orchestration with history-aware hallucination tracking.

Tracks hallucination consistency across multiple conversation turns by
maintaining conversation history and entity memory to detect when the model
contradicts itself across exchanges.
"""

import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from src.utils.logger import get_logger

logger = get_logger(__name__)


class MultiTurnRAGPipeline:
    """Multi-turn RAG pipeline with history-aware hallucination detection.

    Tracks consistency across conversation turns to detect contradictions and
    hallucinations in multi-turn dialogues.

    Args:
        retriever: Retriever instance for document retrieval.
        generator: LLMGenerator instance for response generation.
        detector: EnsembleDetector or detection model for hallucination scoring.
        router: ConfidenceAdaptiveRouter or mitigation router.
    """

    def __init__(
        self,
        retriever=None,
        generator=None,
        detector=None,
        router=None,
    ) -> None:
        self.retriever = retriever
        self.generator = generator
        self.detector = detector
        self.router = router
        self.conversation_history: List[Dict[str, Any]] = []
        self.entity_memory: Dict[str, List[str]] = {}
        self.turn_counter: int = 0
        self.logger = logging.getLogger(__name__)

        logger.info("MultiTurnRAGPipeline initialized.")

    def _extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract entities using regex. Returns dict of entity_type -> List[value]"""
        entities: Dict[str, List[str]] = {}
        entities["years"] = re.findall(r"\b(?:19|20)\d{2}\b", text)
        entities["numbers"] = re.findall(r"\b\d+\.?\d*\b", text)
        months = (
            "January|February|March|April|May|June|July|August|September|October|November|December"
        )
        entities["dates"] = re.findall(rf"\b(?:{months})\b", text)
        entities["proper_nouns"] = re.findall(r"\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\b", text)
        return entities

    def _check_entity_consistency(self, current_entities: Dict[str, List[str]]) -> Tuple[float, List[str]]:
        """Compare current entities against entity_memory.

        Returns:
            (turn_consistency_score: float, conflicts: List[str])
        """
        conflicts: List[str] = []
        if not self.entity_memory:
            return 1.0, []

        type_scores = []
        for entity_type, current_values in current_entities.items():
            past_values = self.entity_memory.get(entity_type, [])
            if not past_values or not current_values:
                continue

            type_conflicts = 0
            for val in current_values:
                if val not in past_values:
                    type_conflicts += 1
                    conflicts.append(
                        f"Entity conflict in '{entity_type}': '{val}' contradicts past values {past_values}"
                    )
            type_score = max(0.0, 1.0 - (type_conflicts / len(current_values)))
            type_scores.append(type_score)

        if not type_scores:
            return 1.0, []

        turn_consistency_score = sum(type_scores) / len(type_scores)
        return turn_consistency_score, conflicts

    def run_turn(self, query: str) -> Dict[str, Any]:
        """Run one conversation turn with history-aware hallucination detection.

        Args:
            query: User question for this turn

        Returns:
            Standard pipeline dict plus:
              turn_id: int
              turn_consistency_score: float (1.0=consistent, 0.0=all conflict)
              entity_conflicts: List[str]
              conversation_length: int
        """
        start_time = time.time()
        self.turn_counter += 1

        # Step 1: Retrieve context
        context_chunks: List[str] = []
        if self.retriever is not None:
            docs = self.retriever.retrieve(query)
            context_chunks = [
                getattr(doc, "page_content", str(doc)) for doc in docs
            ]

        # Step 2: Generate response
        raw_response = ""
        if self.generator is not None:
            raw_response = self.generator.generate(query=query, context_chunks=context_chunks)
        final_response = raw_response

        # Step 3: Extract entities from response
        current_entities = self._extract_entities(raw_response)

        # Step 4: Check consistency against entity_memory
        if self.turn_counter == 1:
            turn_consistency_score = 1.0
            conflicts: List[str] = []
        else:
            turn_consistency_score, conflicts = self._check_entity_consistency(current_entities)

        # Step 5: Update entity_memory with current entities
        for entity_type, values in current_entities.items():
            if entity_type not in self.entity_memory:
                self.entity_memory[entity_type] = []
            for v in values:
                if v not in self.entity_memory[entity_type]:
                    self.entity_memory[entity_type].append(v)

        # Step 6: Run ensemble detector on response alone
        detection_result: Dict[str, Any] = {}
        ensemble_score = 0.0
        if self.detector is not None:
            try:
                detection_result = self.detector.score(
                    query=query,
                    response=raw_response,
                    context_chunks=context_chunks,
                )
                if isinstance(detection_result, dict):
                    ensemble_score = detection_result.get("final_score", 0.0)
                elif isinstance(detection_result, (int, float)):
                    ensemble_score = float(detection_result)
            except Exception as e:
                self.logger.error("Detector scoring error in run_turn: %s", e)
                ensemble_score = 0.5
                detection_result = {"error": str(e), "final_score": 0.5}

        # Step 7: Compute final score
        if self.turn_counter == 1:
            final_score = ensemble_score
        else:
            final_score = 0.7 * ensemble_score + 0.3 * (1.0 - turn_consistency_score)

        is_hallucinated = final_score > 0.5
        mitigation_applied = "none"
        mitigation_details = None

        # Step 8: Run router if hallucination detected
        if is_hallucinated and self.router is not None:
            try:
                if hasattr(self.router, "route_and_mitigate"):
                    router_result = self.router.route_and_mitigate(
                        query=query,
                        response=raw_response,
                        context_chunks=context_chunks,
                        hallucination_score=final_score,
                        detection_result=detection_result,
                    )
                elif hasattr(self.router, "route"):
                    router_result = self.router.route(
                        query=query,
                        response=raw_response,
                        context_chunks=context_chunks,
                        hallucination_score=final_score,
                        detection_result=detection_result,
                    )
                else:
                    router_result = {}

                if isinstance(router_result, dict):
                    final_response = router_result.get("final_response", final_response)
                    mitigation_applied = router_result.get("strategy_used", "applied")
                    mitigation_details = router_result
            except Exception as e:
                self.logger.error("Router error in run_turn: %s", e)

        latency_ms = round((time.time() - start_time) * 1000, 2)

        result_dict: Dict[str, Any] = {
            "query": query,
            "retrieved_chunks": context_chunks,
            "raw_response": raw_response,
            "hallucination_score": final_score,
            "is_hallucinated": is_hallucinated,
            "hallucination_detected": is_hallucinated,
            "final_response": final_response,
            "mitigation_applied": mitigation_applied,
            "mitigation_details": mitigation_details,
            "detection_breakdown": detection_result,
            "latency_ms": latency_ms,
            "turn_id": self.turn_counter,
            "turn_consistency_score": turn_consistency_score,
            "entity_conflicts": conflicts,
            "conversation_length": self.turn_counter,
        }

        self.conversation_history.append(result_dict)
        return result_dict

    def reset(self) -> None:
        """Clear all history for new conversation session."""
        self.conversation_history = []
        self.entity_memory = {}
        self.turn_counter = 0

    def get_conversation_summary(self) -> Dict[str, Any]:
        """Return summary of entire conversation."""
        total_turns = len(self.conversation_history)
        if total_turns == 0:
            return {
                "total_turns": 0,
                "mean_hallucination_score": 0.0,
                "turns_with_entity_conflicts": 0,
                "all_detected_entities": {},
            }

        scores = [t.get("hallucination_score", 0.0) for t in self.conversation_history]
        mean_score = sum(scores) / total_turns
        conflict_turns = sum(
            1 for t in self.conversation_history if t.get("entity_conflicts")
        )

        return {
            "total_turns": total_turns,
            "mean_hallucination_score": round(mean_score, 4),
            "turns_with_entity_conflicts": conflict_turns,
            "all_detected_entities": dict(self.entity_memory),
        }
