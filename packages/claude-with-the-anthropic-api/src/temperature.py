from dotenv import load_dotenv

from anthropic import Anthropic
from anthropic.types import TextBlock

load_dotenv()


client = Anthropic()
model = "claude-haiku-4-5"

message = client.messages.create(
    model=model,
    max_tokens=1000,
    messages=[{"role": "user", "content": "A beautiful day."}],
    temperature=0,
)

for content in message.content:
    if isinstance(content, TextBlock):
        print(content.text)
