from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()


client = Anthropic()
model = "claude-haiku-4-5"


with client.messages.stream(
    model=model,
    max_tokens=1000,
    messages=[
        {
            "role": "user",
            "content": "Generate three different sample AWS CLI commands. Each should be very short.",
        },
        {
            "role": "assistant",
            "content": "```bash",
        },
    ],
    stop_sequences=["```"],
) as stream:
    for text in stream.text_stream:
        print(text, end="")
