"""LLM generator wrapper using Groq API for RAG response generation.

Provides a configurable LLM generation interface with retry logic,
token usage tracking, and a faithfulness-enforcing system prompt
designed for grounded RAG responses.
"""

import logging
import time
from typing import List, Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_SYSTEM_PROMPT = (
    "You are a precise and helpful assistant. Answer ONLY based on the provided "
    "context. If the answer is not in the context, say 'I cannot find this in the "
    "provided documents.' Always cite which part of the context supports your answer. "
    "Do not add information beyond what is explicitly stated in the context."
)


class LLMGenerator:
    """LLM response generator using Groq API.

    Wraps the Groq Python SDK for fast LLM inference with configurable
    model selection, temperature, and automatic retry logic with
    exponential backoff.

    Args:
        model_name: Groq model identifier (default: llama3-8b-8192).
        temperature: Sampling temperature (default: 0.1 for deterministic output).
        max_tokens: Maximum tokens in generated response (default: 512).
        max_retries: Number of retry attempts on API failure (default: 3).
        base_delay: Base delay in seconds for exponential backoff (default: 1.0).
    """

    def __init__(
        self,
        model_name: str = "openai/gpt-oss-20b",
        temperature: float = 0.1,
        max_tokens: int = 512,
        max_retries: int = 3,
        base_delay: float = 1.0,
    ) -> None:
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.base_delay = base_delay

        # Token usage tracking
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_calls = 0

        # Lazy-loaded client
        self._client = None

        logger.info(
            "LLMGenerator initialized (model=%s, temp=%.2f, max_tokens=%d)",
            model_name, temperature, max_tokens,
        )

    @property
    def client(self):
        """Lazy-load the Groq client."""
        if self._client is None:
            import os

            from dotenv import load_dotenv

            load_dotenv()

            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                raise ValueError(
                    "GROQ_API_KEY not found in environment variables. "
                    "Set it in your .env file or export it directly."
                )

            from groq import Groq

            self._client = Groq(api_key=api_key)
            logger.info("Groq client initialized.")

        return self._client

    def generate(
        self,
        query: str,
        context_chunks: List[str],
        system_prompt: Optional[str] = None,
    ) -> str:
        """Generate a response grounded in the provided context chunks.

        Constructs a prompt combining the system instruction, context chunks,
        and user query, then sends it to the Groq API with retry logic.

        Args:
            query: The user's question or instruction.
            context_chunks: List of relevant context passages to ground the response.
            system_prompt: Custom system prompt. Uses DEFAULT_SYSTEM_PROMPT if None.

        Returns:
            Generated response text from the LLM.

        Raises:
            RuntimeError: If all retry attempts fail.
        """
        if system_prompt is None:
            system_prompt = DEFAULT_SYSTEM_PROMPT

        # Build the context-augmented prompt
        user_message = self._build_prompt(query, context_chunks)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        # Retry loop with exponential backoff
        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )

                # Track token usage
                if response.usage:
                    self.total_prompt_tokens += response.usage.prompt_tokens
                    self.total_completion_tokens += response.usage.completion_tokens
                    self.total_calls += 1
                    logger.debug(
                        "Token usage — prompt: %d, completion: %d",
                        response.usage.prompt_tokens,
                        response.usage.completion_tokens,
                    )

                result = response.choices[0].message.content
                return result.strip() if result else ""

            except Exception as e:
                last_error = e
                delay = self.base_delay * (2 ** attempt)
                logger.warning(
                    "API call failed (attempt %d/%d): %s. Retrying in %.1fs...",
                    attempt + 1, self.max_retries, str(e), delay,
                )
                time.sleep(delay)

        error_msg = (
            f"All {self.max_retries} retry attempts failed. Last error: {last_error}"
        )
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    def _build_prompt(self, query: str, context_chunks: List[str]) -> str:
        """Build the context-augmented user prompt.

        Args:
            query: User's question.
            context_chunks: Retrieved context passages.

        Returns:
            Formatted prompt string combining context and query.
        """
        if context_chunks:
            context_text = "\n\n---\n\n".join(
                f"[Context {i+1}]:\n{chunk}" for i, chunk in enumerate(context_chunks)
            )
            prompt = (
                f"Context:\n{context_text}\n\n"
                f"---\n\n"
                f"Question: {query}\n\n"
                f"Answer based ONLY on the context above:"
            )
        else:
            prompt = query

        return prompt

    def get_usage_stats(self) -> dict:
        """Get cumulative token usage statistics.

        Returns:
            Dictionary with total_prompt_tokens, total_completion_tokens,
            total_calls, and average_tokens_per_call.
        """
        avg = (
            (self.total_prompt_tokens + self.total_completion_tokens) / self.total_calls
            if self.total_calls > 0
            else 0
        )
        return {
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_calls": self.total_calls,
            "average_tokens_per_call": round(avg, 1),
        }
