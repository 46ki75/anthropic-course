import pytest
from anthropic import omit
from anthropic.types import MessageParam, TextBlock

import util
from util import add_assistant_message, add_user_message, chat


def test_add_user_message_appends_role_and_content():
    messages: list[MessageParam] = []
    add_user_message(messages, "hello")
    assert messages == [{"role": "user", "content": "hello"}]


def test_add_assistant_message_appends_role_and_content():
    messages: list[MessageParam] = []
    add_assistant_message(messages, "hi there")
    assert messages == [{"role": "assistant", "content": "hi there"}]


def test_messages_accumulate_in_order():
    messages: list[MessageParam] = []
    add_user_message(messages, "q1")
    add_assistant_message(messages, "a1")
    add_user_message(messages, "q2")
    assert messages == [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"},
    ]


class _StubResponse:
    def __init__(self, text: str):
        self.content = [TextBlock(type="text", text=text, citations=None)]


class _StubMessages:
    def __init__(self, text: str):
        self._text = text
        self.last_kwargs: dict[str, object] = {}

    def create(self, **kwargs: object):
        self.last_kwargs = kwargs
        return _StubResponse(self._text)


class _StubClient:
    def __init__(self, text: str):
        self.messages = _StubMessages(text)


@pytest.fixture
def stub_client(monkeypatch: pytest.MonkeyPatch):
    client = _StubClient("stubbed reply")
    monkeypatch.setattr(util, "client", client)
    return client


def test_chat_returns_first_text_block(stub_client: _StubClient):
    messages: list[MessageParam] = [{"role": "user", "content": "hi"}]
    assert chat(messages) == "stubbed reply"


def test_chat_omits_system_and_stop_sequences_by_default(stub_client: _StubClient):
    chat([{"role": "user", "content": "hi"}])
    kwargs = stub_client.messages.last_kwargs
    assert kwargs["system"] is omit
    assert kwargs["stop_sequences"] is omit
    assert kwargs["temperature"] == 1.0


def test_chat_forwards_system_and_stop_sequences(stub_client: _StubClient):
    chat(
        [{"role": "user", "content": "hi"}],
        system="be terse",
        temperature=0.2,
        stop_sequences=["```"],
    )
    kwargs = stub_client.messages.last_kwargs
    assert kwargs["system"] == "be terse"
    assert kwargs["stop_sequences"] == ["```"]
    assert kwargs["temperature"] == 0.2


def test_chat_raises_when_first_block_is_not_text(monkeypatch: pytest.MonkeyPatch):
    class _NonTextResponse:
        content = [object()]

    class _NonTextMessages:
        def create(self, **_: object):
            return _NonTextResponse()

    class _NonTextClient:
        messages = _NonTextMessages()

    monkeypatch.setattr(util, "client", _NonTextClient())
    with pytest.raises(RuntimeError, match="Expected TextBlock"):
        chat([{"role": "user", "content": "hi"}])
