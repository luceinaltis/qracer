"""GeminiAdapter — LLM provider for Google Gemini models.

Maps qracer roles to appropriate Gemini models:
- researcher → gemini-2.0-flash (fast, cost-effective for data gathering)
- analyst    → gemini-2.5-pro (deep analysis)
- strategist → gemini-2.5-pro (investment decisions)
- reporter   → gemini-2.0-flash (summaries, low cost)
"""

from __future__ import annotations

from qracer.llm.providers import (
    CompletionRequest,
    CompletionResponse,
    Role,
)

try:
    from google import generativeai as genai  # pyright: ignore[reportMissingImports]

    _HAS_GENAI = True
except ImportError:
    _HAS_GENAI = False

DEFAULT_MODEL_MAP: dict[Role, str] = {
    Role.RESEARCHER: "gemini-2.0-flash",
    Role.ANALYST: "gemini-2.5-pro",
    Role.STRATEGIST: "gemini-2.5-pro",
    Role.REPORTER: "gemini-2.0-flash",
}

# Approximate per-token costs (USD) — input / output per 1M tokens
_MODEL_COSTS: dict[str, tuple[float, float]] = {
    "gemini-2.5-pro": (1.25, 10.00),
    "gemini-2.0-flash": (0.10, 0.40),
}


def _estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate cost in USD for a completion."""
    input_rate, output_rate = _MODEL_COSTS.get(model, (0.0, 0.0))
    return (input_tokens * input_rate + output_tokens * output_rate) / 1_000_000


class GeminiAdapter:
    """LLM adapter for Google Gemini models."""

    roles: list[Role] = list(Role)

    def __init__(
        self,
        api_key: str | None = None,
        model_map: dict[Role, str] | None = None,
    ) -> None:
        if not _HAS_GENAI:
            raise ImportError(
                "google-generativeai is not installed. Install it with: uv add google-generativeai"
            )
        self._api_key = api_key
        if api_key:
            genai.configure(api_key=api_key)
        self._model_map = model_map or dict(DEFAULT_MODEL_MAP)

    def model_for_role(self, role: Role) -> str:
        """Get the model assigned to a role."""
        return self._model_map.get(role, DEFAULT_MODEL_MAP[Role.RESEARCHER])

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        """Send a completion request to Gemini."""
        model_name = request.model or self.model_for_role(Role.RESEARCHER)

        # Separate system instruction from conversation messages
        system_text: str | None = None
        contents: list[dict[str, str]] = []
        for msg in request.messages:
            if msg.role == "system":
                system_text = msg.content
            else:
                # Gemini uses "user" and "model" (not "assistant")
                role = "model" if msg.role == "assistant" else "user"
                contents.append({"role": role, "parts": msg.content})

        config: dict[str, object] = {
            "max_output_tokens": request.max_tokens,
            "temperature": request.temperature,
        }

        model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=system_text,
            generation_config=config,
        )

        response = await model.generate_content_async(contents)

        content = response.text or ""
        input_tokens = 0
        output_tokens = 0
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            input_tokens = getattr(response.usage_metadata, "prompt_token_count", 0) or 0
            output_tokens = getattr(response.usage_metadata, "candidates_token_count", 0) or 0

        cost = _estimate_cost(model_name, input_tokens, output_tokens)

        return CompletionResponse(
            content=content,
            model=model_name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost,
        )
