# https://anthropic.skilljar.com/claude-with-the-anthropic-api/287727

from anthropic.types import MessageParam

from util import add_assistant_message, add_user_message, chat


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
