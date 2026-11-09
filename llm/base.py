from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import ClassVar

from schemas import ChatMessage, GenerationParams, ModelResponse


class LLMClientBase(ABC):
    # Environment variable holding the provider's API key; checked by the manager before connecting.
    api_key_env: ClassVar[str]

    def __init__(self, model: str, max_retries: int | None = None):
        self.model = model
        # Options forwarded to the SDK client; omitted ones keep the SDK defaults.
        self._sdk_options = {"max_retries": max_retries} if max_retries is not None else {}

    @abstractmethod
    async def chat(
        self, messages: list[ChatMessage], params: GenerationParams | None = None
    ) -> ModelResponse:
        """Send the conversation to the provider and return a normalized response."""

    @abstractmethod
    def generate(
        self, messages: list[ChatMessage], params: GenerationParams | None = None
    ) -> AsyncIterator[str]:
        """Stream the response text chunk by chunk as the provider generates it."""

    async def aclose(self) -> None:
        # Subclasses store their SDK async client in self._client.
        await self._client.close()
