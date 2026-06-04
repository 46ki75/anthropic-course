from typing import Iterable

from dotenv import load_dotenv
from anthropic import Anthropic, omit
from anthropic.types import (
    MessageParam,
    TextBlockParam,
    ToolUnionParam,
    Message,
    ContentBlockParam,
)

MessageContent = str | Iterable[ContentBlockParam]

load_dotenv()


client = Anthropic()
model = "claude-haiku-4-5"


def add_user_message(messages: list[MessageParam], message: MessageContent | Message):
    content = message.content if isinstance(message, Message) else message
    messages.append({"role": "user", "content": content})


def add_assistant_message(
    messages: list[MessageParam], message: MessageContent | Message
):
    content = message.content if isinstance(message, Message) else message
    messages.append({"role": "assistant", "content": content})


def chat(
    messages: list[MessageParam],
    system: str | Iterable[TextBlockParam] | None = None,
    temperature: float = 1.0,
    stop_sequences: list[str] = [],
    tools: list[ToolUnionParam] = [],
) -> Message:
    message = client.messages.create(
        model=model,
        max_tokens=1000,
        messages=messages,
        temperature=temperature,
        system=system if system is not None else omit,
        stop_sequences=stop_sequences if stop_sequences else omit,
        tools=tools,
    )
    return message


def text_from_message(message: Message) -> str:
    return "\n".join([block.text for block in message.content if block.type == "text"])
