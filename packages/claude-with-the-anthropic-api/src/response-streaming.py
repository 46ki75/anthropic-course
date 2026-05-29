from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()


client = Anthropic()
model = "claude-haiku-4-5"


with client.messages.stream(
    model=model,
    max_tokens=1000,
    messages=[
        {"role": "user", "content": "What is quantum computing? Answer in one sentence"}
    ],
) as stream:
    for text in stream.text_stream:
        print(text, end="")
