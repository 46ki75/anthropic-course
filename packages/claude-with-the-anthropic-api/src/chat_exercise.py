# https://anthropic.skilljar.com/claude-with-the-anthropic-api/287727

from dotenv import load_dotenv

load_dotenv()

from anthropic import Anthropic
from anthropic.types import MessageParam

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
    return str(message.content[0].text)


messages: list[MessageParam] = []


def exit():
    print()
    print("Bye!")


def handle_input(line: str):
    add_user_message(messages, f"{line!r}")
    response = chat(messages)
    add_assistant_message(messages, response)
    print()
    print("| Assistant |")
    print(response)
    print()


def repl():
    while True:

        try:
            print("| User |")
            line = input(">>> ")
        except (EOFError, KeyboardInterrupt):
            exit()
            break

        line = line.strip()

        if not line:
            continue

        if line in ("exit", "quit"):
            exit()
            break

        handle_input(line)


if __name__ == "__main__":
    repl()
