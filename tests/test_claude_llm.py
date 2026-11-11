"""ClaudeLLM tests with a stubbed client: checks the request shape and response handling
without network access."""
from types import SimpleNamespace

import pytest

from jobfit.llm import ClaudeLLM, RefusalError
from jobfit.models import RequirementList


class StubMessages:
    def __init__(self, response):
        self.response = response
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return self.response


def make_llm(text, stop_reason="end_turn"):
    response = SimpleNamespace(
        stop_reason=stop_reason,
        content=[SimpleNamespace(type="text", text=text)],
        _request_id="req_test",
    )
    messages = StubMessages(response)
    client = SimpleNamespace(beta=SimpleNamespace(messages=messages))
    return ClaudeLLM(client=client), messages


def test_extract_uses_structured_output_and_fallbacks():
    llm, messages = make_llm('{"requirements": [{"text": "Python", "importance": "must"}]}')
    result = llm.extract_requirements("We need Python")
    assert isinstance(result, RequirementList)
    assert result.requirements[0].text == "Python"
    kw = messages.kwargs
    assert kw["model"] == "claude-opus-5-5"
    assert kw["output_config"]["format"]["type"] == "json_schema"
    assert kw["output_config"]["effort"] == "medium"
    assert kw["fallbacks"] == "default"
    assert "<posting>" in kw["messages"][0]["content"]


def test_refusal_raises():
    llm, _ = make_llm("", stop_reason="refusal")
    with pytest.raises(RefusalError):
        llm.extract_requirements("posting")
