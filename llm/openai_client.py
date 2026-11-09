from collections.abc import AsyncIterator

import openai

from llm.base import LLMClientBase
from schemas import ChatMessage, GenerationParams, ModelResponse


class OpenAIClient(LLMClientBase):
    api_key_env = "OPENAI_API_KEY"

    def __init__(self, model: str, max_retries: int | None = None):
        super().__init__(model, max_retries)
        self._client = openai.AsyncOpenAI(**self._sdk_options)

    def _build_request(self, messages: list[ChatMessage], params: GenerationParams | None) -> dict:
        params = params or GenerationParams()
        sampling = params.model_dump(include={"temperature", "top_p"}, exclude_none=True)
        return {
            "model": self.model,
            "messages": [m.model_dump() for m in messages],
            "max_completion_tokens": params.max_tokens,
            **sampling,
        }

    async def chat(
        self, messages: list[ChatMessage], params: GenerationParams | None = None
    ) -> ModelResponse:
        response = await self._client.chat.completions.create(**self._build_request(messages, params))
        choice = response.choices[0]
        return ModelResponse(
            content=choice.message.content or "",
            model=response.model,
            input_tokens=response.usage.prompt_tokens if response.usage else None,
            output_tokens=response.usage.completion_tokens if response.usage else None,
            stop_reason=choice.finish_reason,
        )

    async def generate(
        self, messages: list[ChatMessage], params: GenerationParams | None = None
    ) -> AsyncIterator[str]:
        stream = await self._client.chat.completions.create(
            **self._build_request(messages, params), stream=True
        )
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
