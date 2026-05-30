# https://anthropic.skilljar.com/claude-with-the-anthropic-api/287735

from anthropic.types import MessageParam

from util import add_assistant_message, add_user_message, chat


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
