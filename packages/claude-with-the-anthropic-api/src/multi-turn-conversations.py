# https://anthropic.skilljar.com/claude-with-the-anthropic-api/287735

from dotenv import load_dotenv

load_dotenv()

from anthropic import Anthropic
from anthropic.types import MessageParam

client = Anthropic()
model = "claude-haiku-4-5"


def add_user_message(messages: list[MessageParam], text):
    user_message: MessageParam = {"role": "user", "content": text}
    messages.append(user_message)


def add_assistant_message(messages: list[MessageParam], text):
    assistant_message: MessageParam = {"role": "assistant", "content": text}
    messages.append(assistant_message)


def chat(messages: list[MessageParam]):
    message = client.messages.create(
        model=model,
        max_tokens=1000,
        messages=messages,
    )
    return message.content[0].text


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
