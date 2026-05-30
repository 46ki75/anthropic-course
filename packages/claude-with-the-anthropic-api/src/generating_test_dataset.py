import json

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
    assert isinstance(block, TextBlock)
    return block.text


def generate_dataset():
    prompt = """
Generate an evaluation dataset for a prompt evaluation. The dataset will be used to evaluate prompts that generate Python, JSON, or Regex specifically for AWS-related tasks. Generate an array of JSON objects, each representing task that requires Python, JSON, or a Regex to complete.

Example output:
```json
[
  {
    "task": "Description of task",
  },
  ...additional
]
```

* Focus on tasks that can be solved by writing a single Python function, a single JSON object, or a single regex
* Focus on tasks that do not require writing much code

Please generate 3 objects.
"""
    messages: list[MessageParam] = []
    add_user_message(messages, prompt)
    add_assistant_message(messages, "```json")
    text = chat(messages, stop_sequences=["```"])
    return json.loads(text)


if __name__ == "__main__":
    dataset = generate_dataset()
    print(dataset)

    with open("dist/dataset.json", "w") as f:
        json.dump(dataset, f, indent=2)
