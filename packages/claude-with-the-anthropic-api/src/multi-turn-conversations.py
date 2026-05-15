# https://anthropic.skilljar.com/claude-with-the-anthropic-api/287735

from dotenv import load_dotenv

load_dotenv()

from anthropic import Anthropic
from anthropic.types import MessageParam, TextBlock

client = Anthropic()
model = "claude-haiku-4-5"


def add_user_message(messages: list[MessageParam], text: str):
    user_message: MessageParam = {"role": "user", "content": text}
    messages.append(user_message)


def add_assistant_message(messages: list[MessageParam], text: str):
    assistant_message: MessageParam = {"role": "assistant", "content": text}
    messages.append(assistant_message)


def chat(messages: list[MessageParam]) -> str:
    message = client.messages.create(
        model=model,
        max_tokens=1000,
        messages=messages,
    )
    block = message.content[0]
    if not isinstance(block, TextBlock):
        raise RuntimeError(f"Expected TextBlock, got {type(block).__name__}")
    return block.text


messages: list[MessageParam] = []

add_user_message(messages, "Define quantum computing in one sentence")

answer = chat(messages)

add_assistant_message(messages, answer)

add_user_message(messages, "Write another sentence")

answer = chat(messages)

add_assistant_message(messages, answer)

for message in messages:
    print(f"\n| {message['role'].capitalize()} |")
    print(message["content"])
