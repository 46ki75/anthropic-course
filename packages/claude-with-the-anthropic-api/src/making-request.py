from dotenv import load_dotenv

load_dotenv()

from anthropic import Anthropic
from anthropic.types import TextBlock

client = Anthropic()
model = "claude-haiku-4-5"

message = client.messages.create(
    model=model,
    max_tokens=1000,
    messages=[
        {"role": "user", "content": "What is quantum computing? Answer in one sentence"}
    ],
)

for content in message.content:
    if isinstance(content, TextBlock):
        print(content.text)
