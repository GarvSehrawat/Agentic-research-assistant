"""LLM provider abstraction.

The rest of the codebase talks to `LLMProvider`, never to a specific
vendor SDK. This keeps the agent, tools, and prompts independent of
which model backend is configured.
"""
from __future__ import annotations

import json
import logging
from typing import Optional, Type, TypeVar

from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class LLMError(Exception):
    """Raised for any LLM configuration or execution failure."""


class LLMProvider:
    """Abstract interface every concrete provider must implement."""

    def generate(self, prompt: str, system: Optional[str] = None) -> str:
        """Return a plain-text completion."""
        raise NotImplementedError

    def generate_structured(
        self,
        prompt: str,
        schema: Type[T],
        system: Optional[str] = None,
        max_retries: int = 1,
    ) -> T:
        """Return a completion validated against a Pydantic schema.

        Default implementation: ask for JSON, parse, validate, and retry
        once with the validation error appended to the prompt if it
        fails. Concrete providers may override this for native
        structured-output support.
        """
        instruction = (
            f"{prompt}\n\n"
            "Respond with ONLY a single valid JSON object matching this "
            f"schema (no prose, no markdown fences):\n{schema.model_json_schema()}"
        )
        last_error: Optional[Exception] = None
        for attempt in range(max_retries + 1):
            raw = self.generate(instruction, system=system)
            cleaned = _strip_code_fences(raw)
            try:
                data = json.loads(cleaned)
                return schema.model_validate(data)
            except (json.JSONDecodeError, ValidationError) as exc:
                last_error = exc
                logger.warning("Structured output validation failed (attempt %s): %s", attempt + 1, exc)
                instruction = (
                    f"{prompt}\n\nYour previous response was invalid JSON or did not "
                    f"match the schema. Error: {exc}\n"
                    f"Respond again with ONLY valid JSON matching this schema:\n"
                    f"{schema.model_json_schema()}"
                )
        raise LLMError(f"LLM did not return valid structured output: {last_error}")


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.startswith("json"):
            text = text[4:]
    return text.strip().strip("`").strip()


class OpenAIProvider(LLMProvider):
    """LLM provider backed by the OpenAI Chat Completions API."""

    def __init__(
    self,
    api_key: Optional[str],
    model: str,
    base_url: Optional[str] = None,
):
        if not api_key:
            raise LLMError(
                "OPENAI_API_KEY is not configured. Set it in your .env file."
            )
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise LLMError(
                "The 'openai' package is not installed. Run: pip install openai"
            ) from exc

        self.model = model
        self._client = OpenAI(
    api_key=api_key,
    base_url=base_url,
)

    def generate(self, prompt: str, system: Optional[str] = None) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.2,
            )
        except Exception as exc:  # noqa: BLE001 - surface as a controlled error
            raise LLMError(f"OpenAI request failed: {exc}") from exc
        return response.choices[0].message.content or ""


def get_llm_provider(config: Optional[dict] = None) -> LLMProvider:
    """Factory that builds the configured LLM provider.

    `config` may be a plain dict (e.g. Flask's `app.config`) with the
    same keys as `app.config.Config`. Falls back to environment
    variables via the Config class defaults when not provided.
    """
    from app.config import Config

    cfg = config or Config.as_dict()
    provider_name = (cfg.get("LLM_PROVIDER") or "openai").lower()
    model = cfg.get("LLM_MODEL") or "gpt-4o-mini"
    api_key = cfg.get("OPENAI_API_KEY")

    if provider_name == "openai":
        return OpenAIProvider(
            api_key=api_key,
            model=model,
        )

    if provider_name == "openrouter":
        return OpenAIProvider(
            api_key=api_key,
            model=model,
            base_url="https://openrouter.ai/api/v1",
        )

    raise LLMError(f"Unsupported LLM_PROVIDER: {provider_name}")
