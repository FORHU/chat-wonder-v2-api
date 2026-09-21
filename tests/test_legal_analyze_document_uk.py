"""cw#80 — analyze_document_uk: the UK sibling of analyze_document.

Unlike the Philippine tool (which has GPT write the analysis and cite G.R. numbers from memory), this returns the
document's text plus UK analysis instructions and no generated analysis, because the UK persona may only state law it has
fetched. No S3, OpenAI or UK Legal MCP call: s3_storage is replaced by a fake module.
"""

import os
import sys
import types
import unittest
from unittest.mock import patch

import the_server as srv

srv._load_user_functions(overwrite_globals=True)
import resources.functions.user_functions as uf  # noqa: E402  (loaded above)

TOOL = "analyze_document_uk"


class _FakeS3:
    """Replaces s3_storage: 'downloads' the given bytes to the requested path, or reports the file as missing."""

    def __init__(self, data=None):
        self.data = data
        self.requested = []

    def __enter__(self):
        fake = types.ModuleType("s3_storage")

        def download_from_s3(key, path):
            self.requested.append((key, path))
            if self.data is None:
                return False
            with open(path, "wb") as f:
                f.write(self.data)
            return True

        fake.download_from_s3 = download_from_s3
        self._patch = patch.dict(sys.modules, {"s3_storage": fake})
        self._patch.start()
        return self

    def __exit__(self, *exc):
        self._patch.stop()


class AnalyzeDocumentUkTests(unittest.TestCase):
    def test_returns_the_text_and_a_uk_guide_with_no_generated_analysis(self):
        with _FakeS3("Tenancy agreement between A and B.".encode()):
            result = uf.analyze_document_uk("uploads/documents/x-lease.txt", "lease.txt")
        self.assertTrue(result["success"])
        self.assertEqual(result["document_text"], "Tenancy agreement between A and B.")
        self.assertEqual(result["filename"], "lease.txt")
        self.assertFalse(result["truncated"])
        self.assertNotIn("ai_summary", result)
        self.assertIn("England & Wales", result["analysis_guide"])
        self.assertIn("citations_resolve", result["analysis_guide"])

    def test_the_guide_forbids_stating_law_that_was_not_fetched(self):
        with _FakeS3(b"text"):
            guide = uf.analyze_document_uk("uploads/x.txt")["analysis_guide"]
        self.assertIn("do not state what an Act or case provides", guide)

    def test_never_calls_an_llm_for_a_text_document(self):
        with _FakeS3(b"text"), patch.dict(sys.modules, {"openai": None}):
            self.assertTrue(uf.analyze_document_uk("uploads/x.txt")["success"])

    def test_filename_defaults_to_the_key_basename(self):
        with _FakeS3(b"text"):
            self.assertEqual(uf.analyze_document_uk("uploads/documents/abc-deed.txt")["filename"], "abc-deed.txt")

    def test_long_text_is_cut_and_flagged(self):
        with _FakeS3(("x" * (uf._UK_ANALYSIS_CHAR_LIMIT + 10)).encode()):
            result = uf.analyze_document_uk("uploads/long.txt")
        self.assertTrue(result["truncated"])
        self.assertEqual(len(result["document_text"]), uf._UK_ANALYSIS_CHAR_LIMIT)
        self.assertEqual(result["char_count"], uf._UK_ANALYSIS_CHAR_LIMIT + 10)
        self.assertIn("covers only the start", result["analysis_guide"])

    def test_latin_1_text_is_decoded(self):
        with _FakeS3("caf\xe9 lease".encode("latin-1")):
            self.assertEqual(uf.analyze_document_uk("uploads/x.txt")["document_text"], "caf\xe9 lease")

    def test_missing_key_is_an_error_and_does_not_touch_s3(self):
        with _FakeS3(b"text") as s3:
            result = uf.analyze_document_uk("")
        self.assertFalse(result["success"])
        self.assertEqual(s3.requested, [])

    def test_file_not_in_s3_is_an_error(self):
        with _FakeS3(None):
            result = uf.analyze_document_uk("uploads/gone.pdf")
        self.assertFalse(result["success"])
        self.assertIn("not found", result["error"])

    def test_oversize_file_is_refused(self):
        with _FakeS3(b"x"), patch.object(uf, "_UK_ANALYSIS_MAX_BYTES", 0):
            result = uf.analyze_document_uk("uploads/big.txt")
        self.assertFalse(result["success"])
        self.assertIn("too large", result["error"])

    def test_empty_document_is_an_error(self):
        with _FakeS3(b"   \n "):
            result = uf.analyze_document_uk("uploads/blank.txt")
        self.assertFalse(result["success"])
        self.assertIn("No text", result["error"])

    def test_unsupported_type_is_an_error(self):
        with _FakeS3(b"data"):
            result = uf.analyze_document_uk("uploads/thing.xyz")
        self.assertFalse(result["success"])
        self.assertIn("Unsupported file type", result["error"])

    def test_image_without_an_openai_key_is_an_error(self):
        with _FakeS3(b"png"), patch.dict(os.environ, {"OPENAI_API_KEY": ""}):
            result = uf.analyze_document_uk("uploads/scan.png")
        self.assertFalse(result["success"])
        self.assertIn("OpenAI API key", result["error"])

    def test_the_temp_file_is_removed(self):
        with _FakeS3(b"text") as s3:
            uf.analyze_document_uk("uploads/cleanup.txt")
        self.assertFalse(os.path.exists(s3.requested[0][1]))


class UkAnalyzeDocumentWiringTests(unittest.TestCase):
    @staticmethod
    def _tool_names(tag):
        _persona, _cleaned, tools, _addendum = srv.process_persona(tag)
        return {t["function"]["name"] for t in (tools or [])}

    def test_uk_persona_has_it_and_the_philippine_persona_does_not(self):
        self.assertIn(TOOL, self._tool_names("[legal ai uk] analyse this"))
        self.assertNotIn(TOOL, self._tool_names("[legal ai] analyze this"))

    def test_the_philippine_tool_is_still_philippine_only(self):
        self.assertIn("analyze_document", self._tool_names("[legal ai] analyze this"))
        self.assertNotIn("analyze_document", self._tool_names("[legal ai uk] analyse this"))

    def test_manifest_entry_requires_the_key_and_says_it_is_not_an_analysis(self):
        tool = next(t["function"] for t in srv._context.all_fun_manifest if t["function"]["name"] == TOOL)
        self.assertEqual(tool["parameters"]["required"], ["s3_key"])
        self.assertIn("NOT an analysis", tool["description"])

    def test_uk_prompt_says_when_to_use_it(self):
        _persona, _cleaned, _tools, addendum = srv.process_persona("[legal ai uk] analyse this")
        self.assertIn(TOOL, addendum)
        self.assertIn("analysis_guide", addendum)

    def test_summary_describe_and_label_use_the_filename(self):
        result = {"filename": "lease.pdf", "char_count": 1234, "truncated": True}
        self.assertEqual(srv._summarize_tool_result(TOOL, result), 'Read "lease.pdf" (1,234 chars (truncated)) for analysis.')
        args = {"s3_key": "uploads/documents/u-lease.pdf"}
        self.assertIn("u-lease.pdf", srv._describe_tool_args(TOOL, __import__("json").dumps(args)))
        self.assertEqual(srv._trace_label_for_call(TOOL, args), "Reading: u-lease.pdf")


if __name__ == "__main__":
    unittest.main()
