import json
import ast
import re
from typing import Literal

from statistics import mean

from anthropic.types import MessageParam
from pydantic import BaseModel

from util import add_user_message, add_assistant_message, chat


class TestCase(BaseModel):
    task: str
    format: str


class TestResult(BaseModel):
    output: str
    test_case: TestCase
    score: float
    reasoning: str


def run_prompt(test_case: TestCase):
    """Merges the prompt and test case input, then returns the result"""
    prompt = f"""
* Respond only with Python, JSON, or a plain Regex
* Do not add any comments or commentary or explanation

Please solve the following task:

{test_case.task}
"""

    messages: list[MessageParam] = []
    add_user_message(messages, prompt)
    add_assistant_message(messages, "```code")
    output = chat(messages, stop_sequences=["```code"])
    return output


def grade_by_model(test_case: TestCase, output: str):
    # Create evaluation prompt
    eval_prompt = f"""
    You are an expert code reviewer. Evaluate this AI-generated solution.

    Task: {test_case.task}
    Solution: {output}

    Provide your evaluation as a structured JSON object with:
    - "strengths": An array of 1-3 key strengths
    - "weaknesses": An array of 1-3 key areas for improvement
    - "reasoning": A concise explanation of your assessment
    - "score": A number between 1-10
    """

    messages: list[MessageParam] = []
    add_user_message(messages, eval_prompt)
    add_assistant_message(messages, "```json")

    eval_text = chat(messages, stop_sequences=["```"])
    return json.loads(eval_text)


def validate_json(text: str) -> Literal[0, 10]:
    try:
        json.loads(text.strip())
        return 10
    except json.JSONDecodeError:
        return 0


def validate_python(text: str) -> Literal[0, 10]:
    try:
        ast.parse(text.strip())
        return 10
    except SyntaxError:
        return 0


def validate_regex(text: str) -> Literal[0, 10]:
    try:
        re.compile(text.strip())
        return 10
    except re.error:
        return 0


def grade_syntax(text: str, test_case: TestCase) -> Literal[0, 10]:
    match test_case.format:
        case "json":
            return validate_json(text)
        case "python":
            return validate_python(text)
        case "regex":
            return validate_regex(text)
        case _:
            return 0


def run_test_case(test_case: TestCase) -> TestResult:
    """Calls run_prompt, then grades the result"""
    output = run_prompt(test_case)

    # Grade the output
    model_grade = grade_by_model(test_case, output)
    model_score = model_grade["score"]
    reasoning = model_grade["reasoning"]

    syntax_score = grade_syntax(output, test_case)

    score = (model_score + syntax_score) / 2

    return TestResult(
        output=output, test_case=test_case, score=score, reasoning=reasoning
    )


def run_eval(dataset: list[TestCase]) -> list[TestResult]:
    """Loads the dataset and calls run_test_case with each case"""
    results: list[TestResult] = []

    for test_case in dataset:
        result = run_test_case(test_case)
        results.append(result)

    average_score = mean([result.score for result in results])
    print(f"Average score: {average_score}")

    return results


if __name__ == "__main__":
    with open("./dist/dataset.json", "r") as f:
        dataset = [TestCase(**item) for item in json.load(f)]

    results = run_eval(dataset)
    print(json.dumps([r.model_dump() for r in results], indent=2))

    with open("dist/eval.json", "w") as f:
        json.dump([r.model_dump() for r in results], f, indent=2)
