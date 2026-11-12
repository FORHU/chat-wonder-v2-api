# -*- coding: utf-8 -*-
"""Title extraction from the Korean Law MCP's plain-text results.

Regression coverage for a live bug: this MCP returns plain formatted text for every tool, not
structured JSON (unlike juris.ph/the UK Legal MCP), so pooled entries had no "title"/"url" key
for _search_results_to_source_metadata to read and showed up in the chat UI's Sources panel as
"Untitled (no url)" even though retrieval worked. Samples below are captured verbatim from a
live call against https://mcp.gomdori.app/law."""

import unittest

from korean_law_mcp.labels import title_for


class KoreanLawLabelsTests(unittest.TestCase):
    def test_search_law_title(self):
        text = (
            "검색 결과 (총 3건):\n\n"
            "📍 정확매칭 (1건):\n"
            "1. 근로기준법 [현행]\n"
            "   - 법령ID: 001872\n"
            "   - MST: 285279\n"
        )
        self.assertEqual(title_for("search_law", {"query": "근로기준법"}, text), "Korean Law Search: 근로기준법")

    def test_get_law_text_title(self):
        text = (
            "법령명: 근로기준법\n"
            "공포일: 20260407\n"
            "시행일: 20261008\n\n"
            "제23조 해고 등의 제한\n"
            "제23조(해고 등의 제한)\n"
            "① 사용자는 근로자에게 정당한 이유 없이...\n"
        )
        self.assertEqual(
            title_for("get_law_text", {"mst": "285279", "jo": "제23조"}, text),
            "근로기준법 제23조 해고 등의 제한",
        )

    def test_search_decisions_title(self):
        text = (
            "판례 검색 결과 (총 408건, 1페이지):\n\n"
            "[616245] 부당해고구제재심판정취소\n"
            "  사건번호: 2023두54914\n"
            "  법원: 대법원\n"
            "  선고일: 20260129\n"
        )
        self.assertEqual(
            title_for("search_decisions", {"domain": "precedent", "query": "부당해고"}, text),
            "부당해고구제재심판정취소 (2023두54914)",
        )

    def test_unknown_tool_falls_back_to_first_line(self):
        self.assertEqual(title_for("some_future_tool", {}, "첫 번째 줄\n두 번째 줄"), "첫 번째 줄")

    def test_empty_text_never_crashes(self):
        for tool in ("search_law", "get_law_text", "get_annexes", "ordinance_radar",
                     "legal_research", "legal_analysis", "search_decisions", "get_decision_text"):
            title_for(tool, {}, "")  # must not raise


if __name__ == "__main__":
    unittest.main()
