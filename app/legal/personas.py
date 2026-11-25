# -*- coding: utf-8 -*-
"""Legal and intake persona resolution."""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Tuple

INTAKE_PERSONAS = ("intake", "intake_kr")

PersonaResult = Tuple[str, str, Optional[List[dict]], Optional[str]]


def _filter(manifest: List[dict], names: List[str]) -> List[dict]:
    return [t for t in manifest if t["function"]["name"] in names]


def resolve_persona(
    user_input: str,
    jurisdiction: Optional[str],
    manifest: List[dict],
) -> PersonaResult:
    """Resolve legal / intake personas. Returns ("auto", ...) when unmatched."""
    persona = "auto"
    filtered_tools = None
    addendum_override = None
    _uk_jurisdiction = (jurisdiction or "").strip().upper() == "UK"
    _kr_jurisdiction = (jurisdiction or "").strip().upper() == "KR"

    _explicit_kr_tag = user_input.lower().startswith("[legal ai kr]")
    if _explicit_kr_tag or (user_input.lower().startswith("[legal ai]") and _kr_jurisdiction):
        persona = "legal_kr"
        user_input = user_input[13:].strip() if _explicit_kr_tag else user_input[10:].strip()
        legal_kr_whitelist = [
            "search_law",
            "get_law_text",
            "get_annexes",
            "ordinance_radar",
            "legal_research",
            "legal_analysis",
            "search_decisions",
            "get_decision_text",
            "discover_tools_kr",
            "execute_tool_kr",
            "get_case_document",
        ]
        filtered_tools = _filter(manifest, legal_kr_whitelist)
        try:
            with open("resources/prompts/legal_prompt_kr.txt", "r", encoding="utf-8") as f:
                addendum_override = f.read()
        except Exception as e:
            logging.error(f"Failed to load legal_prompt_kr.txt: {e}")
        return persona, user_input, filtered_tools, addendum_override

    _explicit_uk_tag = user_input.lower().startswith("[legal ai uk]")
    if _explicit_uk_tag or (user_input.lower().startswith("[legal ai]") and _uk_jurisdiction):
        persona = "legal_uk"
        user_input = user_input[13:].strip() if _explicit_uk_tag else user_input[10:].strip()
        legal_uk_whitelist = [
            "case_law_search",
            "judgment_get_header",
            "judgment_get_index",
            "judgment_get_paragraph",
            "case_law_grep_judgment",
            "legislation_search",
            "legislation_get_toc",
            "legislation_get_section",
            "get_legal_recommendation_uk",
            "citations_parse",
            "citations_resolve",
            "citations_network",
            "citations_format_oscola",
            "analyze_document_uk",
            "parliament_search_hansard",
            "parliament_policy_position_summary",
            "parliament_find_member",
            "parliament_member_debates",
            "parliament_member_interests",
            "parliament_search_petitions",
            "parliament_get_debate_divisions",
            "parliament_get_debate_contributions",
            "parliament_lookup_by_column",
            "bills_search_bills",
            "bills_get_bill",
            "votes_search_divisions",
            "votes_get_division",
            "committees_search_committees",
            "committees_get_committee",
            "committees_search_evidence",
            "hmrc_get_vat_rate",
            "hmrc_check_mtd_status",
            "hmrc_search_guidance",
            "uk_legal_mcp_list_prompts",
            "uk_legal_mcp_get_prompt",
            "uk_legal_mcp_list_resources",
            "uk_legal_mcp_read_resource",
            "get_case_document",
            "generate_legal_document_uk",
            "draft_pleading_uk",
        ]
        filtered_tools = _filter(manifest, legal_uk_whitelist)
        try:
            with open("resources/prompts/legal_prompt_uk.txt", "r", encoding="utf-8") as f:
                addendum_override = f.read()
        except Exception as e:
            logging.error(f"Failed to load legal_prompt_uk.txt: {e}")
        return persona, user_input, filtered_tools, addendum_override

    if user_input.lower().startswith("[legal ai]"):
        persona = "legal"
        user_input = user_input[10:].strip()
        legal_whitelist = [
            "search_jurisprudence",
            "search_republic_acts",
            "get_case",
            "get_republic_act",
            "get_case_document",
            "get_legal_recommendation",
            "analyze_document",
            "generate_legal_document",
            "draft_pleading",
        ]
        filtered_tools = _filter(manifest, legal_whitelist)
        try:
            with open("resources/prompts/legal_prompt.txt", "r", encoding="utf-8") as f:
                addendum_override = f.read()
        except Exception as e:
            logging.error(f"Failed to load legal_prompt.txt: {e}")
        return persona, user_input, filtered_tools, addendum_override

    if user_input.lower().startswith("[intake kr]") or (
        user_input.lower().startswith("[intake]") and _kr_jurisdiction
    ):
        persona = "intake_kr"
        user_input = (
            user_input[11:].strip()
            if user_input.lower().startswith("[intake kr]")
            else user_input[8:].strip()
        )
        intake_kr_whitelist = [
            "search_law",
            "get_law_text",
            "search_decisions",
            "get_decision_text",
            "mark_case_ready",
            "create_case",
        ]
        filtered_tools = _filter(manifest, intake_kr_whitelist)
        try:
            with open("resources/prompts/intake_prompt_kr.txt", "r", encoding="utf-8") as f:
                addendum_override = f.read()
        except Exception as e:
            logging.error(f"Failed to load intake_prompt_kr.txt: {e}")
        return persona, user_input, filtered_tools, addendum_override

    if user_input.lower().startswith("[intake]"):
        persona = "intake"
        user_input = user_input[8:].strip()
        intake_whitelist = [
            "case_law_search",
            "judgment_get_header",
            "legislation_search",
            "legislation_get_section",
            "find_lawyer",
            "mark_case_ready",
            "create_case",
        ]
        filtered_tools = _filter(manifest, intake_whitelist)
        try:
            with open("resources/prompts/intake_prompt.txt", "r", encoding="utf-8") as f:
                addendum_override = f.read()
        except Exception as e:
            logging.error(f"Failed to load intake_prompt.txt: {e}")
        return persona, user_input, filtered_tools, addendum_override

    return "auto", user_input, None, None
