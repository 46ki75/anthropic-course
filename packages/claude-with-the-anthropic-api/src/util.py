from typing import Iterable

from dotenv import load_dotenv
from anthropic import Anthropic, omit
from anthropic.types import MessageParam, TextBlock, TextBlockParam

load_dotenv()


client = Anthropic()
model = "claude-haiku-4-5"


def add_user_message(messages: list[MessageParam], text: str):
    user_message: MessageParam = {"role": "user", "content": text}
    messages.append(user_message)


def add_assistant_message(messages: list[MessageParam], text: str):
    assistant_message: MessageParam = {"role": "assistant", "content": text}
    messages.append(assistant_message)


def chat(
    messages: list[MessageParam],
    system: str | Iterable[TextBlockParam] | None = None,
    temperature: float = 1.0,
    stop_sequences: list[str] = [],
) -> str:
    response = client.messages.create(
        model=model,
        max_tokens=1000,
        messages=messages,
        temperature=temperature,
        system=system if system is not None else omit,
        stop_sequences=stop_sequences if stop_sequences else omit,
    )
    block = response.content[0]
    if not isinstance(block, TextBlock):
        raise RuntimeError(f"Expected TextBlock, got {type(block).__name__}")
    return block.text
