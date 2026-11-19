"""model_registry.py: vendor dispatch (is_deepseek_model/get_client/shape_chat_args) and the
ModelSpec capability lookup added for CW-4/CW-5 (chat-wonder-v2-api#149's model-routing plan,
step 1). No test file existed for this module before; this backfills the existing behavior too,
not just the new addition.
"""

import os
import unittest
from unittest.mock import patch

import model_registry as mr


class IsDeepseekModelTests(unittest.TestCase):
    def test_recognizes_deepseek_models_case_insensitively(self):
        self.assertTrue(mr.is_deepseek_model("deepseek-chat"))
        self.assertTrue(mr.is_deepseek_model("DeepSeek-Reasoner"))

    def test_rejects_openai_models_and_empty_input(self):
        self.assertFalse(mr.is_deepseek_model("gpt-5.6-luna"))
        self.assertFalse(mr.is_deepseek_model(""))
        self.assertFalse(mr.is_deepseek_model(None))


class SupportsLogprobsTests(unittest.TestCase):
    def test_allowlisted_prefixes_only(self):
        self.assertTrue(mr.supports_logprobs("gpt-4o"))
        self.assertTrue(mr.supports_logprobs("gpt-4o-mini"))
        self.assertTrue(mr.supports_logprobs("gpt-4.1"))
        self.assertTrue(mr.supports_logprobs("deepseek-chat"))

    def test_responses_only_and_reasoner_models_are_excluded(self):
        # gpt-5.6-luna/terra (Responses API, no logprobs) and deepseek-reasoner (hard error on
        # logprobs per DeepSeek's docs) must never pass -- confidence.py depends on this gap.
        self.assertFalse(mr.supports_logprobs("gpt-5.6-luna"))
        self.assertFalse(mr.supports_logprobs("gpt-5.6-terra"))
        self.assertFalse(mr.supports_logprobs("deepseek-reasoner"))


class GetClientTests(unittest.TestCase):
    def test_non_deepseek_model_reuses_the_fallback_client(self):
        fallback = object()
        self.assertIs(mr.get_client("gpt-4o", fallback), fallback)

    def test_deepseek_model_without_an_api_key_raises(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(RuntimeError):
                mr.get_client("deepseek-chat", object())

    def test_deepseek_model_builds_and_caches_a_client(self):
        mr._deepseek_client = None
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "k"}), patch("model_registry.make_openai_client") as make:
            make.return_value = "the-client"
            first = mr.get_client("deepseek-chat", object())
            second = mr.get_client("deepseek-reasoner", object())
        self.assertEqual(first, "the-client")
        self.assertIs(first, second)
        make.assert_called_once()
        mr._deepseek_client = None


class ShapeChatArgsTests(unittest.TestCase):
    def test_deepseek_forces_non_thinking_and_drops_reasoning_effort(self):
        out = mr.shape_chat_args("deepseek-chat", {"reasoning_effort": "high", "extra_body": {"foo": "bar"}})
        self.assertNotIn("reasoning_effort", out)
        self.assertEqual(out["extra_body"], {"foo": "bar", "thinking": {"type": "disabled"}})

    def test_strips_logprobs_outside_the_allowlist_regardless_of_vendor(self):
        out = mr.shape_chat_args("gpt-5.6-luna", {"logprobs": True, "top_logprobs": 5})
        self.assertNotIn("logprobs", out)
        self.assertNotIn("top_logprobs", out)

    def test_leaves_logprobs_for_an_allowlisted_model(self):
        out = mr.shape_chat_args("gpt-4o", {"logprobs": True})
        self.assertTrue(out["logprobs"])

    def test_non_deepseek_model_is_otherwise_untouched(self):
        args = {"temperature": 0.2, "reasoning_effort": "low"}
        out = mr.shape_chat_args("gpt-4o", dict(args))
        self.assertEqual(out["temperature"], 0.2)
        self.assertEqual(out["reasoning_effort"], "low")


class ModelSpecTests(unittest.TestCase):
    def test_unknown_model_gets_the_default_no_known_quirks(self):
        spec = mr.model_spec("some-future-model")
        self.assertFalse(spec.requires_responses_api_with_reasoning_tools)
        self.assertTrue(spec.supports_temperature_on_responses)
        self.assertTrue(spec.supports_reasoning_effort)

    def test_empty_or_none_model_also_gets_the_default(self):
        self.assertIs(mr.model_spec(""), mr._DEFAULT_MODEL_SPEC)
        self.assertIs(mr.model_spec(None), mr._DEFAULT_MODEL_SPEC)

    def test_lookup_is_case_insensitive(self):
        self.assertIs(mr.model_spec("GPT-5.6-Luna"), mr.model_spec("gpt-5.6-luna"))

    def test_gpt_5_6_luna_needs_responses_api_for_tools_with_reasoning(self):
        spec = mr.model_spec("gpt-5.6-luna")
        self.assertTrue(spec.requires_responses_api_with_reasoning_tools)
        self.assertTrue(spec.supports_temperature_on_responses)  # only terra lost temperature, not luna

    def test_gpt_5_6_terra_also_cannot_take_temperature_on_responses(self):
        spec = mr.model_spec("gpt-5.6-terra")
        self.assertTrue(spec.requires_responses_api_with_reasoning_tools)
        self.assertFalse(spec.supports_temperature_on_responses)

    def test_deepseek_models_do_not_support_reasoning_effort(self):
        self.assertFalse(mr.model_spec("deepseek-chat").supports_reasoning_effort)
        self.assertFalse(mr.model_spec("deepseek-reasoner").supports_reasoning_effort)


if __name__ == "__main__":
    unittest.main()
