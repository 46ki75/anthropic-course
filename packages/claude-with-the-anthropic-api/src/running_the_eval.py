import json
from typing import Iterable

from dotenv import load_dotenv
from anthropic import Anthropic, omit
from anthropic.types import MessageParam, TextBlock, TextBlockParam


from pydantic import BaseModel

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


class TestCase(BaseModel):
    task: str


class TestResult(BaseModel):
    output: str
    test_case: TestCase
    score: int


def run_prompt(test_case: TestCase):
    """Merges the prompt and test case input, then returns the result"""
    prompt = f"""
Please solve the following task:

{test_case.task}
"""

    messages: list[MessageParam] = []
    add_user_message(messages, prompt)
    output = chat(messages)
    return output


def run_test_case(test_case: TestCase) -> TestResult:
    """Calls run_prompt, then grades the result"""
    output = run_prompt(test_case)

    # TODO - Grading
    score = 10

    return TestResult(output=output, test_case=test_case, score=score)


def run_eval(dataset: list[TestCase]) -> list[TestResult]:
    """Loads the dataset and calls run_test_case with each case"""
    results: list[TestResult] = []

    for test_case in dataset:
        result = run_test_case(test_case)
        results.append(result)

    return results


if __name__ == "__main__":
    with open("./dist/dataset.json", "r") as f:
        dataset = [TestCase(**item) for item in json.load(f)]

    results = run_eval(dataset)
    print(json.dumps([r.model_dump() for r in results], indent=2))
