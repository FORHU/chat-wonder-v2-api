"""Unit tests for the model-string-driven provider resolver."""

import os
import unittest
from unittest import mock

from llm_provider import (
    build_client,
    client_kwargs_for_model,
    is_reasoning_class,
    is_truncated_empty,
    make_think_tag_stream_filter,
    normalize_request_kwargs,
    provider_for_model,
    strip_think_tags,
)


class _FakeChoice:
    def __init__(self, content, finish_reason):
        self.message = mock.Mock(content=content)
        self.finish_reason = finish_reason


class _FakeResponse:
    def __init__(self, content, finish_reason):
        self.choices = [_FakeChoice(content, finish_reason)]


class ProviderForModelTests(unittest.TestCase):
    def test_openai_models(self):
        self.assertEqual(provider_for_model("gpt-4o"), "openai")
        self.assertEqual(provider_for_model("gpt-5.6-terra"), "openai")

    def test_deepseek_models(self):
        self.assertEqual(provider_for_model("deepseek-chat"), "deepseek")
        self.assertEqual(provider_for_model("deepseek-reasoner"), "deepseek")
        self.assertEqual(provider_for_model("deepseek-v4-pro"), "deepseek")
        self.assertEqual(provider_for_model("deepseek-v4-flash"), "deepseek")

    def test_unknown_or_empty_defaults_to_openai(self):
        self.assertEqual(provider_for_model(None), "openai")
        self.assertEqual(provider_for_model(""), "openai")
        self.assertEqual(provider_for_model("some-other-model"), "openai")

    def test_case_insensitive(self):
        self.assertEqual(provider_for_model("DeepSeek-Chat"), "deepseek")


class ClientKwargsForModelTests(unittest.TestCase):
    @mock.patch.dict(os.environ, {"OPENAI_API_KEY": "openai-key"}, clear=True)
    def test_openai_has_no_base_url(self):
        kwargs = client_kwargs_for_model("gpt-4o")
        self.assertEqual(kwargs["api_key"], "openai-key")
        self.assertNotIn("base_url", kwargs)

    @mock.patch.dict(os.environ, {"DEEPSEEK_API_KEY": "ds-key"}, clear=True)
    def test_deepseek_has_base_url_and_key(self):
        kwargs = client_kwargs_for_model("deepseek-chat")
        self.assertEqual(kwargs["api_key"], "ds-key")
        self.assertEqual(kwargs["base_url"], "https://api.deepseek.com/v1")

    @mock.patch.dict(
        os.environ,
        {"DEEPSEEK_API_KEY": "ds-key", "DEEPSEEK_BASE_URL": "https://custom.example/v1"},
        clear=True,
    )
    def test_deepseek_base_url_override(self):
        kwargs = client_kwargs_for_model("deepseek-v4-pro")
        self.assertEqual(kwargs["base_url"], "https://custom.example/v1")


class IsReasoningClassTests(unittest.TestCase):
    def test_reasoning_class_models(self):
        self.assertTrue(is_reasoning_class("gpt-5.6-terra"))
        self.assertTrue(is_reasoning_class("deepseek-reasoner"))
        self.assertTrue(is_reasoning_class("deepseek-v4-pro"))

    def test_non_reasoning_class_models(self):
        self.assertFalse(is_reasoning_class("gpt-4o-mini"))
        self.assertFalse(is_reasoning_class("deepseek-v4-flash"))
        self.assertFalse(is_reasoning_class("deepseek-chat"))

    def test_none_or_empty(self):
        self.assertFalse(is_reasoning_class(None))
        self.assertFalse(is_reasoning_class(""))


class NormalizeRequestKwargsTests(unittest.TestCase):
    def test_n_dropped_only_for_deepseek(self):
        out = normalize_request_kwargs("deepseek-chat", {"n": 1})
        self.assertNotIn("n", out)
        out = normalize_request_kwargs("gpt-4o", {"n": 1})
        self.assertIn("n", out)

    def test_stop_none_dropped_for_both(self):
        out = normalize_request_kwargs("gpt-4o", {"stop": None})
        self.assertNotIn("stop", out)
        out = normalize_request_kwargs("deepseek-chat", {"stop": None})
        self.assertNotIn("stop", out)

    def test_stop_value_kept(self):
        out = normalize_request_kwargs("gpt-4o", {"stop": ["END"]})
        self.assertEqual(out["stop"], ["END"])

    def test_temperature_dropped_when_reasoning_class(self):
        out = normalize_request_kwargs("deepseek-v4-pro", {"temperature": 0.7})
        self.assertNotIn("temperature", out)
        out = normalize_request_kwargs("gpt-5.6-terra", {"temperature": 0.7})
        self.assertNotIn("temperature", out)

    def test_temperature_kept_when_not_reasoning_class(self):
        out = normalize_request_kwargs("deepseek-v4-flash", {"temperature": 0.7})
        self.assertEqual(out["temperature"], 0.7)

    def test_reasoning_effort_dropped_for_deepseek(self):
        out = normalize_request_kwargs("deepseek-v4-pro", {"reasoning_effort": "high"})
        self.assertNotIn("reasoning_effort", out)

    def test_reasoning_effort_kept_for_openai(self):
        out = normalize_request_kwargs("gpt-5.6-terra", {"reasoning_effort": "high"})
        self.assertEqual(out["reasoning_effort"], "high")

    def test_parallel_tool_calls_forced_false_iff_has_tools(self):
        out = normalize_request_kwargs("gpt-4o", {}, has_tools=True)
        self.assertEqual(out["parallel_tool_calls"], False)
        out = normalize_request_kwargs("gpt-4o", {}, has_tools=False)
        self.assertNotIn("parallel_tool_calls", out)

    def test_does_not_mutate_input(self):
        original = {"n": 1}
        normalize_request_kwargs("deepseek-chat", original)
        self.assertEqual(original, {"n": 1})


class BuildClientTests(unittest.TestCase):
    @mock.patch.dict(os.environ, {"OPENAI_API_KEY": "openai-key"}, clear=True)
    def test_builds_openai_client(self):
        client = build_client("gpt-4o")
        self.assertEqual(client.api_key, "openai-key")

    @mock.patch.dict(os.environ, {"DEEPSEEK_API_KEY": "ds-key"}, clear=True)
    def test_builds_deepseek_client(self):
        client = build_client("deepseek-chat")
        self.assertEqual(client.api_key, "ds-key")
        self.assertEqual(str(client.base_url), "https://api.deepseek.com/v1/")


class StripThinkTagsTests(unittest.TestCase):
    def test_no_op_without_tags(self):
        self.assertEqual(strip_think_tags("hello world"), "hello world")

    def test_empty_string(self):
        self.assertEqual(strip_think_tags(""), "")

    def test_strips_single_block(self):
        self.assertEqual(
            strip_think_tags("before <think>internal reasoning</think> after"),
            "before  after",
        )

    def test_strips_multiple_blocks(self):
        self.assertEqual(
            strip_think_tags("<think>a</think>mid<think>b</think>"),
            "mid",
        )

    def test_strips_multiline_block(self):
        text = "start <think>line1\nline2\nline3</think> end"
        self.assertEqual(strip_think_tags(text), "start  end")


class ThinkTagStreamFilterTests(unittest.TestCase):
    def test_no_op_without_tags(self):
        feed = make_think_tag_stream_filter()
        self.assertEqual(feed("hello "), "hello ")
        self.assertEqual(feed("world"), "world")

    def test_strips_single_chunk_block(self):
        feed = make_think_tag_stream_filter()
        result = feed("before <think>reasoning</think> after")
        self.assertEqual(result, "before  after")

    def test_strips_block_split_across_chunks(self):
        feed = make_think_tag_stream_filter()
        chunks = ["before <thi", "nk>internal ", "reasoning</th", "ink> after"]
        result = "".join(feed(c) for c in chunks)
        self.assertEqual(result, "before  after")

    def test_preserves_surrounding_text_across_many_small_chunks(self):
        feed = make_think_tag_stream_filter()
        text = "abc<think>hidden</think>def"
        result = "".join(feed(ch) for ch in text)
        self.assertEqual(result, "abcdef")

    def test_multiple_blocks_in_stream(self):
        feed = make_think_tag_stream_filter()
        chunks = ["a<think>x</think>b", "<think>y</think>c"]
        result = "".join(feed(c) for c in chunks)
        self.assertEqual(result, "abc")

    def test_open_tag_split_at_boundary(self):
        feed = make_think_tag_stream_filter()
        result = "".join(
            feed(c) for c in ["no tags here <thi", "nk>hidden</think> tail"]
        )
        self.assertEqual(result, "no tags here  tail")


class IsTruncatedEmptyTests(unittest.TestCase):
    def test_length_finish_with_empty_content_is_truncated(self):
        response = _FakeResponse(content="", finish_reason="length")
        self.assertTrue(is_truncated_empty(response))

    def test_length_finish_with_none_content_is_truncated(self):
        response = _FakeResponse(content=None, finish_reason="length")
        self.assertTrue(is_truncated_empty(response))

    def test_length_finish_with_whitespace_only_content_is_truncated(self):
        response = _FakeResponse(content="   \n  ", finish_reason="length")
        self.assertTrue(is_truncated_empty(response))

    def test_length_finish_with_real_content_is_not_truncated(self):
        response = _FakeResponse(content="a real answer", finish_reason="length")
        self.assertFalse(is_truncated_empty(response))

    def test_stop_finish_with_empty_content_is_not_truncated(self):
        # Not this function's problem to flag — a model can legitimately answer "".
        response = _FakeResponse(content="", finish_reason="stop")
        self.assertFalse(is_truncated_empty(response))

    def test_malformed_response_returns_false(self):
        self.assertFalse(is_truncated_empty(mock.Mock(choices=[])))
        self.assertFalse(is_truncated_empty(object()))


if __name__ == "__main__":
    unittest.main()
