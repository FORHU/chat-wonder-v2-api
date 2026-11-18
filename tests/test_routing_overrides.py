"""routing_overrides.py (CW-8): the thin, read-only polling client for ilovelawyer-api's
persona routing overrides. No decide_turn_model/decide_tool_model exists yet to call
get_override() for real -- these tests exercise the cache/fail-open/env-fallback contract on
its own, ahead of model_router.py landing.
"""

import json
import os
import unittest
import urllib.error
from unittest.mock import patch

import routing_overrides as ro


def _fake_urlopen(rows):
    class _Resp:
        def __init__(self, payload):
            self._payload = json.dumps(payload).encode("utf-8")

        def read(self):
            return self._payload

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def _opener(req, timeout=10):
        return _Resp({"overrides": rows})

    return _opener


class GetOverrideTests(unittest.TestCase):
    def setUp(self):
        ro._reset_for_tests()
        self.env = patch.dict(os.environ, {"ILOVELAWYER_API_BASE": "http://ilovelawyer-api.test", "CHAT_WONDER_API_KEY": "secret"})
        self.env.start()

    def tearDown(self):
        self.env.stop()
        ro._reset_for_tests()

    def test_fetches_and_returns_a_cached_row(self):
        rows = [{"persona": "legal_uk", "routingMode": "MANUAL", "manualPinnedModel": "gpt-6-luna", "toolOverrides": None}]
        with patch("urllib.request.urlopen", _fake_urlopen(rows)):
            result = ro.get_override("legal_uk")
        self.assertEqual(result["routingMode"], "MANUAL")
        self.assertEqual(result["manualPinnedModel"], "gpt-6-luna")
        self.assertEqual(result["source"], "cache")

    def test_does_not_refetch_within_the_ttl(self):
        rows = [{"persona": "legal_uk", "routingMode": "AUTO", "manualPinnedModel": None, "toolOverrides": None}]
        calls = {"n": 0}

        def _opener(req, timeout=10):
            calls["n"] += 1
            return _fake_urlopen(rows)(req, timeout)

        with patch("urllib.request.urlopen", _opener), patch.dict(os.environ, {"ROUTING_OVERRIDE_CACHE_TTL_S": "9999"}):
            ro.get_override("legal_uk")
            ro.get_override("legal_uk")
            ro.get_override("stylist")
        self.assertEqual(calls["n"], 1)

    def test_refetches_once_the_ttl_has_elapsed(self):
        rows = [{"persona": "legal_uk", "routingMode": "AUTO", "manualPinnedModel": None, "toolOverrides": None}]
        calls = {"n": 0}

        def _opener(req, timeout=10):
            calls["n"] += 1
            return _fake_urlopen(rows)(req, timeout)

        with patch("urllib.request.urlopen", _opener), patch.dict(os.environ, {"ROUTING_OVERRIDE_CACHE_TTL_S": "0"}):
            ro.get_override("legal_uk")
            ro.get_override("legal_uk")
        self.assertEqual(calls["n"], 2)

    def test_a_persona_with_no_row_falls_back_to_its_env_vars(self):
        with patch("urllib.request.urlopen", _fake_urlopen([])), patch.dict(os.environ, {"STYLIST_ROUTING_MODE": "MANUAL", "STYLIST_MANUAL_MODEL": "gpt-4o"}):
            result = ro.get_override("stylist")
        self.assertEqual(result["source"], "env")
        self.assertEqual(result["routingMode"], "MANUAL")
        self.assertEqual(result["manualPinnedModel"], "gpt-4o")

    def test_a_persona_with_no_env_override_defaults_to_auto(self):
        with patch("urllib.request.urlopen", _fake_urlopen([])):
            result = ro.get_override("some_unconfigured_persona")
        self.assertEqual(result["routingMode"], "AUTO")
        self.assertIsNone(result["manualPinnedModel"])

    def test_a_failed_refresh_keeps_serving_the_last_known_snapshot(self):
        rows = [{"persona": "legal_uk", "routingMode": "MANUAL", "manualPinnedModel": "gpt-6-luna", "toolOverrides": None}]
        with patch("urllib.request.urlopen", _fake_urlopen(rows)), patch.dict(os.environ, {"ROUTING_OVERRIDE_CACHE_TTL_S": "0"}):
            first = ro.get_override("legal_uk")
        self.assertEqual(first["source"], "cache")

        def _boom(req, timeout=10):
            raise urllib.error.URLError("connection refused")

        with patch("urllib.request.urlopen", _boom), patch.dict(os.environ, {"ROUTING_OVERRIDE_CACHE_TTL_S": "0"}):
            second = ro.get_override("legal_uk")
        self.assertEqual(second["source"], "cache")
        self.assertEqual(second["manualPinnedModel"], "gpt-6-luna")
        self.assertIsNotNone(ro.last_refresh_error())

    def test_never_having_reached_ilovelawyer_api_falls_back_to_env_for_every_persona(self):
        def _boom(req, timeout=10):
            raise urllib.error.URLError("connection refused")

        with patch("urllib.request.urlopen", _boom):
            result = ro.get_override("legal_uk")
        self.assertEqual(result["source"], "env")
        self.assertEqual(result["routingMode"], "AUTO")

    def test_missing_config_fails_open_to_env_instead_of_raising(self):
        with patch.dict(os.environ, {}, clear=True):
            result = ro.get_override("legal_uk")
        self.assertEqual(result["source"], "env")

    def test_invalid_tool_overrides_json_is_ignored_not_raised(self):
        with patch("urllib.request.urlopen", _fake_urlopen([])), patch.dict(os.environ, {"STYLIST_TOOL_MODEL_OVERRIDES": "not json"}):
            result = ro.get_override("stylist")
        self.assertIsNone(result["toolOverrides"])


if __name__ == "__main__":
    unittest.main()
