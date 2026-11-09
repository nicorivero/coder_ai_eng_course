from typing import Literal

from pydantic import BaseModel, Field

# "system" is kept here for both providers; Anthropic clients must move it to the `system` param.
Role = Literal["system", "user", "assistant"]

Provider = Literal["openai", "anthropic"]


class LLMConfig(BaseModel):
    provider: Provider
    model: str = Field(min_length=1)
    max_retries: int | None = Field(default=None, ge=0)  # None keeps the SDK default


class ChatMessage(BaseModel):
    role: Role
    content: str = Field(min_length=1)


class GenerationParams(BaseModel):
    # None means "don't send it": the provider default applies.
    temperature: float | None = Field(default=None, ge=0, le=2)
    top_p: float | None = Field(default=None, ge=0, le=1)
    max_tokens: int = Field(default=1024, gt=0)


class ModelResponse(BaseModel):
    content: str
    model: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    stop_reason: str | None = None


class LLMError(BaseModel):
    kind: Literal["auth", "rate_limit", "connection", "api"]
    message: str
    status_code: int | None = None
    retryable: bool
