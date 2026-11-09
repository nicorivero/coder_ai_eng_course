import os
from collections.abc import AsyncIterator

import anthropic
import openai

from llm.anthropic_client import AnthropicClient
from llm.base import LLMClientBase
from llm.openai_client import OpenAIClient
from schemas import ChatMessage, GenerationParams, LLMConfig, LLMError, ModelResponse, Provider

_REGISTRY: dict[Provider, type[LLMClientBase]] = {
    "openai": OpenAIClient,
    "anthropic": AnthropicClient,
}

_SDK_ERRORS = (openai.APIError, anthropic.APIError)


def _to_llm_error(exc: Exception) -> LLMError:
    """Map an SDK exception (raised after the SDK's own retries) to a structured LLMError."""
    status = getattr(exc, "status_code", None)
    if status is None:
        kind = "connection"
    elif status in (401, 403):
        kind = "auth"
    elif status == 429:
        kind = "rate_limit"
    else:
        kind = "api"

    return LLMError(
        kind=kind,
        message=getattr(exc, "message", None) or str(exc),
        status_code=status,
        retryable=kind in ("rate_limit", "connection") or (status is not None and status >= 500),
    )


class AsyncLLMManager:
    """Single entry point to the LLM providers; use it with `async with` to close connections."""

    def __init__(self, config: LLMConfig):
        client_cls = _REGISTRY[config.provider]
        if not os.getenv(client_cls.api_key_env):
            raise ValueError(f"{client_cls.api_key_env} is not set in the environment or .env file.")

        self.provider = config.provider
        self._client = client_cls(config.model, config.max_retries)

    @property
    def model(self) -> str:
        return self._client.model

    async def chat(
        self, messages: list[ChatMessage], params: GenerationParams | None = None
    ) -> ModelResponse | LLMError:
        try:
            return await self._client.chat(messages, params)
        except _SDK_ERRORS as e:
            return _to_llm_error(e)

    async def generate(
        self, messages: list[ChatMessage], params: GenerationParams | None = None
    ) -> AsyncIterator[str | LLMError]:
        try:
            async for piece in self._client.generate(messages, params):
                yield piece
        except _SDK_ERRORS as e:
            # Partial text was already yielded; report the failure as the last item.
            yield _to_llm_error(e)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "AsyncLLMManager":
        return self

    async def __aexit__(self, *exc_info) -> None:
        await self.aclose()
