"""OpenAIAdapter — LLM provider for OpenAI GPT models.

Maps qracer roles to appropriate GPT models:
- researcher → gpt-4o-mini (fast, cost-effective for data gathering)
- analyst    → gpt-4o (deep analysis)
- strategist → gpt-4o (investment decisions)
- reporter   → gpt-4o-mini (summaries, low cost)
"""

from __future__ import annotations

from qracer.llm.providers import (
    CompletionRequest,
    CompletionResponse,
    Role,
)

try:
    import openai  # pyright: ignore[reportMissingImports]

    _HAS_OPENAI = True
except ImportError:
    _HAS_OPENAI = False

DEFAULT_MODEL_MAP: dict[Role, str] = {
    Role.RESEARCHER: "gpt-4o-mini",
    Role.ANALYST: "gpt-4o",
    Role.STRATEGIST: "gpt-4o",
    Role.REPORTER: "gpt-4o-mini",
}

# Approximate per-token costs (USD) — input / output per 1M tokens
_MODEL_COSTS: dict[str, tuple[float, float]] = {
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
}


def _estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate cost in USD for a completion."""
    input_rate, output_rate = _MODEL_COSTS.get(model, (0.0, 0.0))
    return (input_tokens * input_rate + output_tokens * output_rate) / 1_000_000


class OpenAIAdapter:
    """LLM adapter for OpenAI GPT models."""

    roles: list[Role] = list(Role)

    def __init__(
        self,
        api_key: str | None = None,
        model_map: dict[Role, str] | None = None,
    ) -> None:
        if not _HAS_OPENAI:
            raise ImportError("openai is not installed. Install it with: uv add openai")
        self._client = openai.AsyncOpenAI(api_key=api_key)
        self._model_map = model_map or dict(DEFAULT_MODEL_MAP)

    def model_for_role(self, role: Role) -> str:
        """Get the model assigned to a role."""
        return self._model_map.get(role, DEFAULT_MODEL_MAP[Role.RESEARCHER])

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        """Send a completion request to OpenAI."""
        model = request.model or self.model_for_role(Role.RESEARCHER)

        # OpenAI uses system as a regular message role
        messages: list[dict[str, str]] = []
        for msg in request.messages:
            messages.append({"role": msg.role, "content": msg.content})

        response = await self._client.chat.completions.create(
            model=model,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            messages=messages,  # type: ignore[arg-type]
        )

        content = ""
        if response.choices:
            content = response.choices[0].message.content or ""

        input_tokens = response.usage.prompt_tokens if response.usage else 0
        output_tokens = response.usage.completion_tokens if response.usage else 0
        cost = _estimate_cost(model, input_tokens, output_tokens)

        return CompletionResponse(
            content=content,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost,
        )
