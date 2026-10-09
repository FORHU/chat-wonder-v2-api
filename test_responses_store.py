# -*- coding: utf-8 -*-
"""Responses API calls never ask OpenAI to keep a copy (issue #116).

The Responses API stores each response provider-side by default. Every call goes through
legal_responses_chain.create_response, which forces store=False; this file checks that, and
fails if a direct responses.create call appears anywhere else in the app.
"""
import re
from pathlib import Path

from legal_responses_chain import create_response

ROOT = Path(__file__).resolve().parent
SKIP_DIRS = {".git", "venv", ".venv", "node_modules", "__pycache__", "tests", "testers", "fixtures"}


class _FakeResponses:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return "response"


class _FakeClient:
    def __init__(self):
        self.responses = _FakeResponses()


def test_store_is_false_when_the_caller_says_nothing():
    client = _FakeClient()
    create_response(client, model="m", input=[{"role": "user", "content": "hi"}], stream=True)
    assert client.responses.calls[0]["store"] is False


def test_a_caller_cannot_switch_storage_back_on():
    client = _FakeClient()
    create_response(client, model="m", input=[], store=True)
    assert client.responses.calls[0]["store"] is False


def test_everything_else_is_passed_through_and_the_result_returned():
    client = _FakeClient()
    result = create_response(client, model="m", input=["x"], stream=True, tools=[1], reasoning={"effort": "high"})
    assert result == "response"
    assert client.responses.calls[0] == {
        "model": "m",
        "input": ["x"],
        "stream": True,
        "tools": [1],
        "reasoning": {"effort": "high"},
        "store": False,
    }


def _python_files():
    for path in ROOT.rglob("*.py"):
        parts = set(path.relative_to(ROOT).parts)
        if parts & SKIP_DIRS or path.name.startswith("test_"):
            continue
        yield path


def test_no_other_code_calls_the_responses_api_directly():
    pattern = re.compile(r"\.responses\.create\b")
    found = []
    for path in _python_files():
        for number, line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
            if pattern.search(line):
                found.append(f"{path.relative_to(ROOT).as_posix()}:{number}")
    # The one allowed call is the one inside create_response.
    assert len(found) == 1 and found[0].startswith("legal_responses_chain.py:"), (
        "Call create_response(client, ...) from legal_responses_chain instead of "
        f"client.responses.create(...); direct calls found at: {found}"
    )
