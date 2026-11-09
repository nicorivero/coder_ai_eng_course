import asyncio
import os

from dotenv import load_dotenv

from llm import AsyncLLMManager
from schemas import ChatMessage, LLMConfig, LLMError

QUESTION = "¿Qué es la entropía?"


def print_error(error: LLMError) -> None:
    print(f"\n[error: {error.kind}, status: {error.status_code}, retryable: {error.retryable}] {error.message}")


async def main() -> None:
    messages = [ChatMessage(role="user", content=QUESTION)]

    load_dotenv()
    try:
        config = LLMConfig(provider=os.getenv("LLM_PROVIDER"), model=os.getenv("LLM_MODEL"))
        llm = AsyncLLMManager(config)
    except ValueError as e:  # pydantic's ValidationError is a ValueError too

        print(f"[config error] {e}")
        return

    async with llm:
        print("=== Normal ===")
        result = await llm.chat(messages)
        if isinstance(result, LLMError):
            print_error(result)
        else:
            print(result.content)
            print(f"[tokens in/out: {result.input_tokens}/{result.output_tokens}, stop: {result.stop_reason}]")

        print("\n=== Streaming ===")
        async for piece in llm.generate(messages):
            if isinstance(piece, LLMError):
                print_error(piece)
            else:
                print(piece, end="", flush=True)
        print()


if __name__ == "__main__":
    asyncio.run(main())
