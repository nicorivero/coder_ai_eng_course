import logging
from collections.abc import AsyncIterator

import anthropic
import httpx2

from llm.base import LLMClientBase
from schemas import ChatMessage, GenerationParams, ModelResponse

logger = logging.getLogger(__name__)


class AnthropicClient(LLMClientBase):
    api_key_env = "ANTHROPIC_API_KEY"

    def __init__(self, model: str, max_retries: int | None = None):
        super().__init__(model, max_retries)
        self._client = anthropic.AsyncAnthropic(**self._sdk_options)

    def _build_request(self, messages: list[ChatMessage], params: GenerationParams | None) -> dict:
        params = params or GenerationParams()
        # Anthropic takes the system prompt as a separate param, not as a message.
        system = "\n\n".join(m.content for m in messages if m.role == "system")
        chat_messages = [m.model_dump() for m in messages if m.role != "system"]
        extra = {"system": system} if system else {}

        # Current Claude models reject sampling params with a 400, so they are dropped.
        if params.temperature is not None or params.top_p is not None:
            logger.warning("temperature/top_p are not supported by Anthropic models and were ignored")

        return {
            "model": self.model,
            "max_tokens": params.max_tokens,
            "messages": chat_messages,
            **extra,
        }

    async def chat(
        self, messages: list[ChatMessage], params: GenerationParams | None = None
    ) -> ModelResponse:
        response = await self._client.messages.create(**self._build_request(messages, params))
        # Keep only text blocks; thinking blocks are not part of the answer.
        content = "".join(block.text for block in response.content if block.type == "text")
        return ModelResponse(
            content=content,
            model=response.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            stop_reason=response.stop_reason,
        )

    async def generate(
        self, messages: list[ChatMessage], params: GenerationParams | None = None
    ) -> AsyncIterator[str]:
        # text_stream yields only answer text, skipping thinking blocks.
        async with self._client.messages.stream(**self._build_request(messages, params)) as stream:
            try:
                async for text in stream.text_stream:
                    yield text
            # The SDK re-raises raw transport errors mid-stream; wrap them like the OpenAI SDK does.
            except httpx2.TimeoutException as e:
                raise anthropic.APITimeoutError(request=stream.response.request) from e
            except httpx2.RequestError as e:
                raise anthropic.APIConnectionError(request=stream.response.request) from e
