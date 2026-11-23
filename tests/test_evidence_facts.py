# tests/test_evidence_facts.py
"""#159: /api/legal/extract-evidence-facts (structured email-screenshot fact pull, falling back
to describe_image's free-text description) and /api/legal/suggest-next-evidence (picks one open
gap to prompt for next). Same mocking shape as test_describe_image.py -- no real OpenAI call, no
live data."""
import asyncio
import unittest
from types import SimpleNamespace as NS
from unittest import mock

import requests
import llm_usage
import the_server as srv


def _fake_http_response(content):
    return NS(content=content, raise_for_status=lambda: None)


def _fake_vision_client(content):
    def create(**kw):
        return NS(model="gpt-4o",
                  usage=NS(prompt_tokens=200, completion_tokens=40, total_tokens=240,
                           prompt_tokens_details=NS(cached_tokens=0)),
                  choices=[NS(message=NS(content=content), finish_reason="stop")])
    return NS(chat=NS(completions=NS(create=create)))


class ExtractEvidenceFactsTests(unittest.TestCase):
    def test_returns_structured_facts_for_an_email_screenshot(self):
        payload = (
            '{"is_email_screenshot": true, '
            '"facts": {"sender": "landlord@example.com", "date": "2026-10-01", '
            '"amount": "£450", "agreement": "agreed to repair the boiler by Friday"}, '
            '"description": null}'
        )
        factory = lambda **kw: llm_usage.UsageTrackingOpenAI(_fake_vision_client(payload))
        req = srv.ExtractEvidenceFactsRequest(image_url="https://example-bucket.s3.amazonaws.com/k/email.png?sig=abc", filename="email.png")
        with mock.patch.object(srv, "make_openai_client", factory), \
                mock.patch.object(srv._context, "openai_api_key", "k"), \
                mock.patch.object(requests, "get", lambda *a, **kw: _fake_http_response(b"fake-png-bytes")):
            out = asyncio.run(srv.extract_evidence_facts(req))
        self.assertTrue(out["success"])
        self.assertTrue(out["is_email_screenshot"])
        self.assertEqual(out["facts"]["sender"], "landlord@example.com")
        self.assertIsNone(out["description"])

    def test_falls_back_to_a_description_for_a_non_email_photo(self):
        payload = '{"is_email_screenshot": false, "facts": null, "description": "A damaged front door with a broken lock."}'
        factory = lambda **kw: llm_usage.UsageTrackingOpenAI(_fake_vision_client(payload))
        req = srv.ExtractEvidenceFactsRequest(image_url="https://example-bucket.s3.amazonaws.com/k/door.jpg?sig=abc", filename="door.jpg")
        with mock.patch.object(srv, "make_openai_client", factory), \
                mock.patch.object(srv._context, "openai_api_key", "k"), \
                mock.patch.object(requests, "get", lambda *a, **kw: _fake_http_response(b"fake-jpg-bytes")):
            out = asyncio.run(srv.extract_evidence_facts(req))
        self.assertTrue(out["success"])
        self.assertFalse(out["is_email_screenshot"])
        self.assertIsNone(out["facts"])
        self.assertIn("door", out["description"])

    def test_missing_key_returns_unsuccessful_without_raising(self):
        req = srv.ExtractEvidenceFactsRequest(image_url="https://example-bucket.s3.amazonaws.com/k/x.png?sig=abc", filename="x.png")
        with mock.patch.object(srv._context, "openai_api_key", None), \
                mock.patch.object(requests, "get", lambda *a, **kw: _fake_http_response(b"bytes")):
            out = asyncio.run(srv.extract_evidence_facts(req))
        self.assertFalse(out["success"])
        self.assertIsNone(out["facts"])
        self.assertIsNone(out["description"])

    def test_missing_image_url_raises_404(self):
        req = srv.ExtractEvidenceFactsRequest(image_url="https://example-bucket.s3.amazonaws.com/missing.png?sig=abc", filename="missing.png")

        def raise_fetch_error(*a, **kw):
            raise Exception("404 Client Error: Not Found")

        with mock.patch.object(requests, "get", raise_fetch_error):
            with self.assertRaises(srv.HTTPException) as ctx:
                asyncio.run(srv.extract_evidence_facts(req))
        self.assertEqual(ctx.exception.status_code, 404)

    def test_oversized_image_skipped_without_calling_openai(self):
        calls = []
        factory = lambda **kw: calls.append(1) or _fake_vision_client("x")
        req = srv.ExtractEvidenceFactsRequest(image_url="https://example-bucket.s3.amazonaws.com/k/huge.png?sig=abc", filename="huge.png")
        oversized = NS(content=b"\x00" * (srv.VISION_MAX_IMAGE_BYTES + 1), raise_for_status=lambda: None)
        with mock.patch.object(srv, "make_openai_client", factory), \
                mock.patch.object(srv._context, "openai_api_key", "k"), \
                mock.patch.object(requests, "get", lambda *a, **kw: oversized):
            out = asyncio.run(srv.extract_evidence_facts(req))
        self.assertFalse(out["success"])
        self.assertEqual(calls, [])

    def test_vision_call_failure_is_non_fatal(self):
        def raise_err(**kw):
            raise RuntimeError("boom")
        factory = lambda **kw: NS(chat=NS(completions=NS(create=raise_err)))
        req = srv.ExtractEvidenceFactsRequest(image_url="https://example-bucket.s3.amazonaws.com/k/x.png?sig=abc", filename="x.png")
        with mock.patch.object(srv, "make_openai_client", factory), \
                mock.patch.object(srv._context, "openai_api_key", "k"), \
                mock.patch.object(requests, "get", lambda *a, **kw: _fake_http_response(b"bytes")):
            out = asyncio.run(srv.extract_evidence_facts(req))
        self.assertFalse(out["success"])


class SuggestNextEvidenceTests(unittest.TestCase):
    def test_no_gaps_returns_no_suggestion_without_calling_openai(self):
        calls = []
        factory = lambda **kw: calls.append(1) or _fake_vision_client("x")
        req = srv.SuggestNextEvidenceRequest(gaps=[])
        with mock.patch.object(srv, "make_openai_client", factory), \
                mock.patch.object(srv._context, "openai_api_key", "k"):
            out = asyncio.run(srv.suggest_next_evidence(req))
        self.assertFalse(out["success"])
        self.assertIsNone(out["suggestion"])
        self.assertEqual(calls, [])

    def test_picks_one_of_the_given_gap_ids(self):
        gaps = [
            srv.EvidenceGap(id="gap-1", label="Proof of payment"),
            srv.EvidenceGap(id="gap-2", label="Tenancy agreement"),
        ]
        payload = '{"gap_id": "gap-2", "message": "Could you upload a copy of your tenancy agreement?"}'
        factory = lambda **kw: _fake_vision_client(payload)
        req = srv.SuggestNextEvidenceRequest(gaps=gaps)
        with mock.patch.object(srv, "make_openai_client", factory), \
                mock.patch.object(srv._context, "openai_api_key", "k"):
            out = asyncio.run(srv.suggest_next_evidence(req))
        self.assertTrue(out["success"])
        self.assertEqual(out["suggestion"]["gapId"], "gap-2")
        self.assertIn("tenancy agreement", out["suggestion"]["message"])

    def test_a_gap_id_not_in_the_given_list_is_rejected(self):
        gaps = [srv.EvidenceGap(id="gap-1", label="Proof of payment")]
        payload = '{"gap_id": "not-a-real-gap", "message": "Upload something."}'
        factory = lambda **kw: _fake_vision_client(payload)
        req = srv.SuggestNextEvidenceRequest(gaps=gaps)
        with mock.patch.object(srv, "make_openai_client", factory), \
                mock.patch.object(srv._context, "openai_api_key", "k"):
            out = asyncio.run(srv.suggest_next_evidence(req))
        self.assertFalse(out["success"])
        self.assertIsNone(out["suggestion"])

    def test_missing_key_returns_unsuccessful_without_raising(self):
        gaps = [srv.EvidenceGap(id="gap-1", label="Proof of payment")]
        req = srv.SuggestNextEvidenceRequest(gaps=gaps)
        with mock.patch.object(srv._context, "openai_api_key", None):
            out = asyncio.run(srv.suggest_next_evidence(req))
        self.assertFalse(out["success"])
        self.assertIsNone(out["suggestion"])


if __name__ == "__main__":
    unittest.main()
