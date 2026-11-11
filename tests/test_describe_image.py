# tests/test_describe_image.py
import asyncio
import unittest
from types import SimpleNamespace as NS
from unittest import mock

import requests
import llm_usage
import the_server as srv


def _fake_http_response(content):
    return NS(content=content, raise_for_status=lambda: None)


def _fake_oversized_http_response():
    return NS(content=b"\x00" * (srv.VISION_MAX_IMAGE_BYTES + 1), raise_for_status=lambda: None)


def _fake_vision_client(content):
    def create(**kw):
        return NS(model="gpt-4o",
                  usage=NS(prompt_tokens=200, completion_tokens=40, total_tokens=240,
                           prompt_tokens_details=NS(cached_tokens=0)),
                  choices=[NS(message=NS(content=content), finish_reason="stop")])
    return NS(chat=NS(completions=NS(create=create)))


class DescribeImageTests(unittest.TestCase):
    def test_returns_description_on_success(self):
        factory = lambda **kw: llm_usage.UsageTrackingOpenAI(_fake_vision_client("A silver sedan with front-end damage, parked on a residential street."))
        req = srv.DescribeImageRequest(image_url="https://example-bucket.s3.amazonaws.com/k/car.jpg?sig=abc", filename="car_accident.jpg")
        fake_content = b"\xff\xd8\xff\xe0fake-jpeg-bytes\xff\xd9"
        with mock.patch.object(srv, "make_openai_client", factory), \
                mock.patch.object(srv._context, "openai_api_key", "k"), \
                mock.patch.object(requests, "get", lambda *a, **kw: _fake_http_response(fake_content)):
            out = asyncio.run(srv.describe_image(req))
        self.assertTrue(out["success"])
        self.assertIn("sedan", out["description"])
        self.assertEqual(out["usage"]["call_count"], 1)

    def test_missing_key_returns_no_description_without_raising(self):
        req = srv.DescribeImageRequest(image_url="https://example-bucket.s3.amazonaws.com/k/car.jpg?sig=abc", filename="car_accident.jpg")
        fake_content = b"\xff\xd8\xff\xe0fake-jpeg-bytes\xff\xd9"
        with mock.patch.object(srv._context, "openai_api_key", None), \
                mock.patch.object(requests, "get", lambda *a, **kw: _fake_http_response(fake_content)):
            out = asyncio.run(srv.describe_image(req))
        self.assertFalse(out["success"])
        self.assertIsNone(out["description"])

    def test_missing_image_url_raises_404(self):
        req = srv.DescribeImageRequest(image_url="https://example-bucket.s3.amazonaws.com/missing.jpg?sig=abc", filename="missing.jpg")

        def raise_fetch_error(*a, **kw):
            raise Exception("404 Client Error: Not Found")

        with mock.patch.object(requests, "get", raise_fetch_error):
            with self.assertRaises(srv.HTTPException) as ctx:
                asyncio.run(srv.describe_image(req))
        self.assertEqual(ctx.exception.status_code, 404)

    def test_oversized_image_skipped_without_calling_openai(self):
        calls = []
        factory = lambda **kw: calls.append(1) or _fake_vision_client("x")
        req = srv.DescribeImageRequest(image_url="https://example-bucket.s3.amazonaws.com/k/huge.jpg?sig=abc", filename="huge.jpg")
        with mock.patch.object(srv, "make_openai_client", factory), \
                mock.patch.object(srv._context, "openai_api_key", "k"), \
                mock.patch.object(requests, "get", lambda *a, **kw: _fake_oversized_http_response()):
            out = asyncio.run(srv.describe_image(req))
        self.assertFalse(out["success"])
        self.assertIsNone(out["description"])
        self.assertEqual(calls, [])
