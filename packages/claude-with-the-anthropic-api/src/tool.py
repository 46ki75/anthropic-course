from datetime import datetime

from pydantic import BaseModel, Field
from anthropic import Anthropic
from anthropic.types import ToolParam, MessageParam

from dotenv import load_dotenv

load_dotenv()


class GetCurrentDatetimeArgs(BaseModel):
    date_format: str = Field(
        default="%Y-%m-%d %H:%M:%S",
        description="A string specifying the format of the returned datetime. "
        "Uses Python's strftime format codes.",
    )


def get_current_datetime(args: GetCurrentDatetimeArgs) -> str:
    if not args.date_format:
        raise ValueError("date_format cannot be empty")
    return datetime.now().strftime(args.date_format)


get_current_datetime_schema = ToolParam(
    {
        "name": "get_current_datetime",
        "description": "Returns the current date and time formatted according to the specified format",
        "input_schema": GetCurrentDatetimeArgs.model_json_schema(),
    }
)

client = Anthropic()
model = "claude-haiku-4-5"

messages: list[MessageParam] = []
messages.append(
    {"role": "user", "content": "What is the exact time, formatted as HH:MM:SS?"}
)


if __name__ == "__main__":
    response = client.messages.create(
        model=model,
        max_tokens=1000,
        messages=messages,
        tools=[get_current_datetime_schema],
    )

    messages.append({"role": "assistant", "content": response.content})

    for content in response.content:
        if content.type == "tool_use":
            if content.name == "get_current_datetime":
                arg = GetCurrentDatetimeArgs.model_validate(content.input)
                result = get_current_datetime(arg)
                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": content.id,
                                "content": result,
                                "is_error": False,
                            }
                        ],
                    }
                )

    response = client.messages.create(
        model=model,
        max_tokens=1000,
        messages=messages,
        tools=[get_current_datetime_schema],
    )
    print(response.content)
