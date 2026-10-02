"""Direct legislation.gov.uk fallback for the UK legal tools.

uk-legal-mcp.fly.dev's legislation_search / legislation_get_section started failing with
"legislation.gov.uk returned an AWS WAF JavaScript challenge" while the same pages answer normally from
our own hosts. The tools now report that as a failure (it used to come back as success=True with the error
text as the "result") and fall back to legislation.gov.uk's public XML endpoints.

The fixtures under tests/fixtures/uk_legislation are trimmed copies of real responses. Network is mocked.
"""

import os
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import resources.functions.user_functions as uf
from uk_legal_mcp import direct
from uk_legal_mcp.client import UkLegalMcpToolError

FIXTURES = Path(__file__).parent / "fixtures" / "uk_legislation"
FEED = (FIXTURES / "feed_title.xml").read_bytes()
SECTION_29 = (FIXTURES / "section_29.xml").read_bytes()

WAF_TEXT = (
    'Internal error: {"error_category": "transient", "is_retryable": true, "attempted": '
    "\"legislation_search(query='x')\", \"description\": \"legislation.gov.uk returned an AWS WAF JavaScript "
    'challenge for https://www.legislation.gov.uk/ukpga/1953/20/section/29."}'
)


class FeedParsingTests(unittest.TestCase):
    def test_old_act_gets_its_calendar_year_and_number_not_the_regnal_ids(self):
        # The feed id for the 1953 Act is .../id/ukpga/Eliz2/1-2/20; the rest of chat-wonder (uk_legal_mcp.urls)
        # needs a 4-digit year, which only the ukm:Year / ukm:Number metadata carries.
        rows = direct.parse_feed(FEED)
        act = next(r for r in rows if r["title"] == "Births and Deaths Registration Act 1953")
        self.assertEqual((act["type"], act["year"], act["number"]), ("ukpga", 1953, 20))
        self.assertEqual(act["url"], "https://www.legislation.gov.uk/ukpga/1953/20")

    def test_modern_act_row(self):
        rows = direct.parse_feed(FEED)
        act = next(r for r in rows if "Scotland" in r["title"])
        self.assertEqual((act["type"], act["year"], act["number"]), ("ukpga", 1965, 49))

    def test_limit_and_year_filter(self):
        self.assertEqual(len(direct.parse_feed(FEED, limit=1)), 1)
        only_1953 = direct.parse_feed(FEED, year=1953)
        self.assertEqual([r["year"] for r in only_1953], [1953])

    def test_empty_feed_is_an_empty_list_not_an_error(self):
        empty = b'<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom"></feed>'
        self.assertEqual(direct.parse_feed(empty), [])


class SearchUrlTests(unittest.TestCase):
    def test_title_search_without_a_type_uses_the_all_feed(self):
        url = direct.search_url("Human Rights Act", None, None, False)
        self.assertTrue(url.startswith("https://www.legislation.gov.uk/all/data.feed?"))
        self.assertIn("title=Human+Rights+Act", url)

    def test_type_and_year_go_in_the_path(self):
        url = direct.search_url("Births", "ukpga", 1953, False)
        self.assertTrue(url.startswith("https://www.legislation.gov.uk/ukpga/1953/data.feed?"))

    def test_fulltext_uses_the_search_feed_with_text(self):
        url = direct.search_url("correction of errors", "ukpga", None, True)
        self.assertTrue(url.startswith("https://www.legislation.gov.uk/search/data.feed?"))
        self.assertIn("text=correction+of+errors", url)
        self.assertIn("type=ukpga", url)


def _feed(*rows):
    """A minimal Atom feed with the given (title, type, year, number) rows."""
    entries = "".join(
        f"<entry><id>http://www.legislation.gov.uk/id/{t}/{y}/{n}</id><title>{title}</title>"
        f'<ukm:Year Value="{y}"/><ukm:Number Value="{n}"/></entry>'
        for title, t, y, n in rows
    )
    return (
        '<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom" '
        f'xmlns:ukm="http://www.legislation.gov.uk/namespaces/metadata">{entries}</feed>'
    ).encode("utf-8")


ACT_1953 = ("Births and Deaths Registration Act 1953", "ukpga", 1953, 20)
EMPTY = _feed()


class QueryReductionTests(unittest.TestCase):
    """The model searches with a sentence or 'Act title + extra words'; legislation.gov.uk's title search only
    matches when the query is part of a title, and its text search needs every word — so both returned 0 rows
    for the real queries (the UI showed '0 results' and no citations)."""

    def test_an_act_title_inside_a_longer_query_is_found(self):
        cands = direct._candidate_titles("Births and Deaths Registration Act 1953 correction error name birth entry")
        self.assertEqual(cands[0], "Births and Deaths Registration Act 1953")

    def test_leading_words_are_dropped_one_at_a_time(self):
        cands = direct._candidate_titles("Under the Human Rights Act 1998 what is the test")
        self.assertIn("Human Rights Act 1998", cands)

    def test_a_bare_act_and_year_is_never_a_candidate(self):
        # 'Act 1953' alone would match every 1953 Act on the site.
        cands = direct._candidate_titles("Births and Deaths Registration Act 1953 correction error")
        self.assertNotIn("Act 1953", cands)
        self.assertTrue(all(len(c.split()) >= 3 for c in cands))
        self.assertEqual(direct._candidate_titles("Equality Act 2010 indirect discrimination")[0], "Equality Act 2010")

    def test_a_plain_sentence_has_no_title_candidates(self):
        self.assertEqual(direct._candidate_titles("Correcting a name recorded on a birth certificate"), [])

    def test_keywords_drop_filler_and_keep_the_subject_words(self):
        kw = direct._keywords(
            "Correcting or changing a name recorded on a birth certificate where the person has used a different "
            "name since childhood"
        )
        self.assertEqual(kw, ["correcting", "changing", "name", "recorded", "birth", "certificate"])

    def test_keywords_are_capped(self):
        self.assertLessEqual(len(direct._keywords("alpha bravo charlie delta echo foxtrot golf hotel")), 6)


class SearchLadderTests(unittest.TestCase):
    def _run(self, responder, query="Births and Deaths Registration Act 1953 correction error name birth entry", **kwargs):
        """`responder(url)` returns the feed bytes for that request (EMPTY if it has nothing)."""
        urls = []

        def fake_fetch(url):
            urls.append(url)
            return responder(url) or EMPTY

        with patch.object(direct, "_fetch", side_effect=fake_fetch):
            return direct.search_legislation(query, **kwargs), urls

    def test_the_act_title_inside_the_query_is_searched_when_the_whole_query_finds_nothing(self):
        def responder(url):  # only the exact Act title matches, as on the real site
            return _feed(ACT_1953) if url.endswith("title=Births+and+Deaths+Registration+Act+1953") else None

        res, urls = self._run(responder)
        self.assertEqual(res["results"][0]["title"], "Births and Deaths Registration Act 1953")
        self.assertEqual(res["strategy"], "act-title")
        self.assertEqual(len(urls), 2)  # whole query (empty), then the title inside it

    def test_a_query_that_is_already_a_title_is_one_request(self):
        res, urls = self._run(lambda url: _feed(ACT_1953), query="Births and Deaths Registration Act 1953")
        self.assertEqual(res["strategy"], "title")
        self.assertEqual(len(urls), 1)

    def test_a_sentence_falls_through_to_or_keywords_on_primary_legislation(self):
        res, urls = self._run(
            lambda url: _feed(ACT_1953) if "search/data.feed" in url else None,
            query="Correcting or changing a name recorded on a birth certificate where the person has used a different name",
        )
        self.assertEqual(res["strategy"], "keywords")
        self.assertEqual(res["results"][0]["title"], "Births and Deaths Registration Act 1953")
        self.assertIn("text=correcting+OR+changing+OR+name+OR+recorded+OR+birth+OR+certificate", urls[-1])
        self.assertIn("type=ukpga", urls[-1])

    def test_keywords_widen_to_all_legislation_when_primary_acts_have_nothing(self):
        si = ("The Registration of Births and Deaths Regulations 1987", "uksi", 1987, 2088)
        res, urls = self._run(
            lambda url: _feed(si) if "search/data.feed" in url and "type=" not in url else None,
            query="registering a stillbirth abroad lately",
        )
        self.assertIn("type=ukpga", urls[-2])  # the ukpga-restricted pass came first and found nothing
        self.assertNotIn("type=", urls[-1])  # the unrestricted pass is the one that answered
        self.assertEqual(res["results"][0]["type"], "uksi")

    def test_a_caller_supplied_type_is_never_widened(self):
        _, urls = self._run(lambda url: None, query="registering a stillbirth abroad lately", type="uksi")
        self.assertTrue(all("type=ukpga" not in u for u in urls))

    def test_nothing_anywhere_is_an_empty_result_not_an_error(self):
        res, _ = self._run(lambda url: None, query="zzzz qqqq xxxx")
        self.assertEqual(res["results"], [])

    def test_fulltext_true_is_respected_first(self):
        res, urls = self._run(lambda url: _feed(ACT_1953) if "search/data.feed" in url else None,
                              query="register correction", fulltext=True)
        self.assertIn("/search/data.feed?", urls[0])
        self.assertEqual(res["strategy"], "fulltext")


class SectionParsingTests(unittest.TestCase):
    def setUp(self):
        self.res = direct.parse_section(SECTION_29, "ukpga", 1953, 20, "29", max_chars=10000)

    def test_heading_number_and_act_title(self):
        self.assertEqual(self.res["title"].strip(), "Correction of errors in registers.")
        self.assertEqual(self.res["section_number"], "29")
        self.assertEqual(self.res["act_title"], "Births and Deaths Registration Act 1953")

    def test_content_is_plain_text_with_the_subsections(self):
        text = self.res["content"]
        self.assertIn("No alteration shall be made in any register of live–births", text)
        self.assertIn("Any clerical error which may from time to time be discovered", text)
        self.assertNotIn("<", text)
        self.assertNotIn("CommentaryRef", text)

    def test_inserted_text_is_kept_in_reading_order(self):
        # Amendments are wrapped in <Addition>; the words must stay where the statute puts them.
        self.assertIn("either by two credible persons having knowledge of the truth of the case", self.res["content"])

    def test_url_is_the_section_page(self):
        self.assertEqual(self.res["url"], "https://www.legislation.gov.uk/ukpga/1953/20/section/29")

    def test_max_chars_truncates_and_says_so(self):
        short = direct.parse_section(SECTION_29, "ukpga", 1953, 20, "29", max_chars=120)
        self.assertLessEqual(len(short["content"]), 120)
        self.assertTrue(short["truncated"])
        self.assertFalse(self.res["truncated"])

    def test_a_document_with_no_section_body_is_an_error(self):
        bare = b'<Legislation xmlns="http://www.legislation.gov.uk/namespaces/legislation"><Primary><Body/></Primary></Legislation>'
        with self.assertRaises(direct.DirectLegislationError):
            direct.parse_section(bare, "ukpga", 1953, 20, "29", max_chars=100)


class FetchTests(unittest.TestCase):
    def _resp(self, status=200, content=b"<x/>"):
        r = MagicMock()
        r.status_code = status
        r.content = content
        r.raise_for_status.side_effect = None if status < 400 else Exception(f"HTTP {status}")
        return r

    def test_request_identifies_itself_and_allows_a_slow_first_fetch(self):
        with patch.object(direct.requests, "get", return_value=self._resp()) as get:
            direct._fetch("https://www.legislation.gov.uk/x")
        _, kwargs = get.call_args
        self.assertIn("chat-wonder", kwargs["headers"]["User-Agent"])
        self.assertGreaterEqual(kwargs["timeout"], 20)  # the first XML fetch took ~6s from our own host

    def test_404_is_reported_as_not_found(self):
        with patch.object(direct.requests, "get", return_value=self._resp(404)):
            with self.assertRaises(direct.DirectLegislationError) as cm:
                direct._fetch("https://www.legislation.gov.uk/missing")
        self.assertIn("not found", str(cm.exception).lower())

    def test_waf_challenge_page_is_reported_not_parsed(self):
        challenge = b"<html><script src='https://x.awswaf.com/challenge.js'></script></html>"
        with patch.object(direct.requests, "get", return_value=self._resp(200, challenge)):
            with self.assertRaises(direct.DirectLegislationError) as cm:
                direct._fetch("https://www.legislation.gov.uk/x")
        self.assertIn("challenge", str(cm.exception).lower())


class McpFailureIsReportedAsFailureTests(unittest.TestCase):
    def _client(self, payload):
        client = MagicMock()
        client.call_tool.return_value = payload
        return client

    def test_is_error_payload_raises_instead_of_looking_like_success(self):
        payload = {"text": WAF_TEXT, "raw_result": {"isError": True, "content": [{"type": "text", "text": WAF_TEXT}]}}
        with patch("uk_legal_mcp.client.get_client", return_value=self._client(payload)):
            with self.assertRaises(UkLegalMcpToolError) as cm:
                uf._uk_call("legislation_search", {"query": "x"})
        self.assertIn("WAF", str(cm.exception))

    def test_a_normal_payload_is_returned_untouched(self):
        payload = {"results": [{"title": "Human Rights Act 1998"}]}
        with patch("uk_legal_mcp.client.get_client", return_value=self._client(payload)):
            self.assertEqual(uf._uk_call("legislation_search", {"query": "x"}), payload)

    def test_every_uk_tool_now_reports_the_failure(self):
        payload = {"text": WAF_TEXT, "raw_result": {"isError": True}}
        with patch("uk_legal_mcp.client.get_client", return_value=self._client(payload)):
            res = uf.legislation_get_toc(type="ukpga", year=1953, number=20)
        self.assertFalse(res["success"])
        self.assertIn("WAF", res["error"])


class FallbackWiringTests(unittest.TestCase):
    ROWS = [{"title": "Births and Deaths Registration Act 1953", "type": "ukpga", "year": 1953, "number": 20,
             "url": "https://www.legislation.gov.uk/ukpga/1953/20"}]
    SECTION = {"title": "Correction of errors in registers.", "section_number": "29", "content": "29 1 No alteration ...",
               "url": "https://www.legislation.gov.uk/ukpga/1953/20/section/29", "truncated": False}

    def _mcp_fails(self):
        return patch.object(uf, "_uk_call", side_effect=UkLegalMcpToolError(WAF_TEXT))

    def test_search_falls_back_to_legislation_gov_uk(self):
        with self._mcp_fails(), patch.object(direct, "search_legislation", return_value={"results": [dict(r) for r in self.ROWS]}) as d:
            res = uf.legislation_search(query="Births and Deaths Registration Act 1953")
        self.assertTrue(res["success"])
        self.assertEqual(res["results"][0]["title"], "Births and Deaths Registration Act 1953")
        self.assertEqual(res["fallback"], "legislation.gov.uk")
        self.assertIn("WAF", res["mcp_error"])
        self.assertIn("score", res["results"][0])  # fill_missing_scores still runs on fallback rows
        d.assert_called_once()

    def test_section_falls_back_to_legislation_gov_uk(self):
        with self._mcp_fails(), patch.object(direct, "get_section", return_value=dict(self.SECTION)) as d:
            res = uf.legislation_get_section(type="ukpga", year=1953, number=20, section="29")
        self.assertTrue(res["success"])
        self.assertEqual(res["section_number"], "29")
        self.assertEqual(res["fallback"], "legislation.gov.uk")
        d.assert_called_once()

    def test_the_mcp_result_is_used_when_it_works(self):
        with patch.object(uf, "_uk_call", return_value={"results": [dict(self.ROWS[0])]}), \
                patch.object(direct, "search_legislation") as d:
            res = uf.legislation_search(query="Births and Deaths Registration Act 1953")
        self.assertTrue(res["success"])
        self.assertNotIn("fallback", res)
        d.assert_not_called()

    def test_both_failing_reports_both_reasons(self):
        with self._mcp_fails(), patch.object(direct, "get_section", side_effect=direct.DirectLegislationError("not found")):
            res = uf.legislation_get_section(type="ukpga", year=1953, number=20, section="99")
        self.assertFalse(res["success"])
        self.assertIn("WAF", res["error"])
        self.assertIn("not found", res["error"])

    def test_fallback_can_be_switched_off(self):
        with self._mcp_fails(), patch.dict(os.environ, {"UK_LEGISLATION_DIRECT_FALLBACK": "false"}), \
                patch.object(direct, "search_legislation") as d:
            res = uf.legislation_search(query="Births and Deaths Registration Act 1953")
        self.assertFalse(res["success"])
        d.assert_not_called()

    def test_the_acronym_is_still_expanded_before_the_direct_search(self):
        with self._mcp_fails(), patch.object(direct, "search_legislation", return_value={"results": []}) as d:
            uf.legislation_search(query="HRA")
        self.assertEqual(d.call_args.args[0] if d.call_args.args else d.call_args.kwargs["query"], "Human Rights Act 1998")


if __name__ == "__main__":
    unittest.main()
