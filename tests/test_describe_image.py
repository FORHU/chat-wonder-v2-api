# tests/test_describe_image.py
import asyncio
import unittest
from types import SimpleNamespace as NS
from unittest import mock

import llm_usage
import the_server as srv


def _write_tmp_jpeg(path):
    with open(path, "wb") as f:
        f.write(b"\xff\xd8\xff\xe0fake-jpeg-bytes\xff\xd9")
    return True


def _write_tmp_oversized_jpeg(path):
    with open(path, "wb") as f:
        f.write(b"\x00" * (srv.VISION_MAX_IMAGE_BYTES + 1))
    return True


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
        req = srv.AnalyzeS3DocumentRequest(s3_key="k/car.jpg", filename="car_accident.jpg")
        with mock.patch.object(srv, "make_openai_client", factory), \
                mock.patch.object(srv._context, "openai_api_key", "k"), \
                mock.patch.object(srv.s3_storage, "download_from_s3", lambda *a, **kw: _write_tmp_jpeg(a[1])):
            out = asyncio.run(srv.describe_image(req))
        self.assertTrue(out["success"])
        self.assertIn("sedan", out["description"])
        self.assertEqual(out["usage"]["call_count"], 1)

    def test_missing_key_returns_no_description_without_raising(self):
        req = srv.AnalyzeS3DocumentRequest(s3_key="k/car.jpg", filename="car_accident.jpg")
        with mock.patch.object(srv._context, "openai_api_key", None), \
                mock.patch.object(srv.s3_storage, "download_from_s3", lambda *a, **kw: _write_tmp_jpeg(a[1])):
            out = asyncio.run(srv.describe_image(req))
        self.assertFalse(out["success"])
        self.assertIsNone(out["description"])

    def test_missing_s3_object_raises_404(self):
        req = srv.AnalyzeS3DocumentRequest(s3_key="missing.jpg", filename="missing.jpg")
        with mock.patch.object(srv.s3_storage, "download_from_s3", lambda *a, **kw: False):
            with self.assertRaises(srv.HTTPException) as ctx:
                asyncio.run(srv.describe_image(req))
        self.assertEqual(ctx.exception.status_code, 404)

    def test_oversized_image_skipped_without_calling_openai(self):
        calls = []
        factory = lambda **kw: calls.append(1) or _fake_vision_client("x")
        req = srv.AnalyzeS3DocumentRequest(s3_key="k/huge.jpg", filename="huge.jpg")
        with mock.patch.object(srv, "make_openai_client", factory), \
                mock.patch.object(srv._context, "openai_api_key", "k"), \
                mock.patch.object(srv.s3_storage, "download_from_s3", lambda *a, **kw: _write_tmp_oversized_jpeg(a[1])):
            out = asyncio.run(srv.describe_image(req))
        self.assertFalse(out["success"])
        self.assertIsNone(out["description"])
        self.assertEqual(calls, [])
