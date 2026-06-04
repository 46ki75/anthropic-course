import json

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Sequence

from pydantic import BaseModel, Field
from anthropic.types import ToolParam, MessageParam, Message, ToolResultBlockParam

from util import chat, add_assistant_message, text_from_message, add_user_message

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class ToolDefinition[ArgsT: BaseModel]:
    schema: ToolParam
    args_model: type[ArgsT]
    handler: Callable[[ArgsT], str]

    @property
    def name(self) -> str:
        return self.schema["name"]

    def run(self, tool_input: object) -> str:
        args = self.args_model.model_validate(tool_input)
        return self.handler(args)


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


get_current_datetime_tool = ToolDefinition(
    schema=ToolParam(
        {
            "name": "get_current_datetime",
            "description": "Returns the current date and time formatted according to the specified format",
            "input_schema": GetCurrentDatetimeArgs.model_json_schema(),
        }
    ),
    args_model=GetCurrentDatetimeArgs,
    handler=get_current_datetime,
)


TOOLS: list[ToolDefinition[Any]] = [
    get_current_datetime_tool,
]


def run_tools(
    message: Message, tools: Sequence[ToolDefinition[Any]]
) -> list[ToolResultBlockParam]:
    tools_by_name = {tool.name: tool for tool in tools}
    tool_requests = [block for block in message.content if block.type == "tool_use"]
    tool_result_blocks: list[ToolResultBlockParam] = []
    for tool_request in tool_requests:
        tool = tools_by_name.get(tool_request.name)
        if tool is None:
            tool_result_blocks.append(
                {
                    "type": "tool_result",
                    "tool_use_id": tool_request.id,
                    "content": f"Error: unknown tool '{tool_request.name}'",
                    "is_error": True,
                }
            )
            continue
        try:
            tool_output = tool.run(tool_request.input)
            tool_result_block: ToolResultBlockParam = {
                "type": "tool_result",
                "tool_use_id": tool_request.id,
                "content": json.dumps(tool_output),
                "is_error": False,
            }
        except Exception as e:
            tool_result_block = {
                "type": "tool_result",
                "tool_use_id": tool_request.id,
                "content": f"Error: {e}",
                "is_error": True,
            }
        tool_result_blocks.append(tool_result_block)
    return tool_result_blocks


messages: list[MessageParam] = []
messages.append(
    {"role": "user", "content": "What is the exact time, formatted as HH:MM:SS?"}
)


if __name__ == "__main__":
    while True:
        response = chat(messages=messages, tools=[tool.schema for tool in TOOLS])
        add_assistant_message(messages, response)
        print(text_from_message(response))

        if response.stop_reason != "tool_use":
            break

        tool_results = run_tools(response, TOOLS)
        add_user_message(messages, tool_results)
