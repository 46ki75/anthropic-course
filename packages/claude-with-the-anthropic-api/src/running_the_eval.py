import json
from statistics import mean

from anthropic.types import MessageParam
from pydantic import BaseModel

from util import add_user_message, add_assistant_message, chat


class TestCase(BaseModel):
    task: str


class TestResult(BaseModel):
    output: str
    test_case: TestCase
    score: int
    reasoning: str


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


def run_test_case(test_case: TestCase) -> TestResult:
    """Calls run_prompt, then grades the result"""
    output = run_prompt(test_case)

    # Grade the output
    model_grade = grade_by_model(test_case, output)
    score = model_grade["score"]
    reasoning = model_grade["reasoning"]

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
