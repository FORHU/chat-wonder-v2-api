import os
import io
import json
import time
import uuid
import hashlib
import tempfile
import logging
import urllib.request
import urllib.parse
import urllib.error
from datetime import datetime, date
from llm_usage import make_openai_client

# Simple in-memory cache with TTL
_search_cache = {}
_logger = logging.getLogger(__name__)
CACHE_TTL_SECONDS = 30






# StyleOS tools live in app.styleos.tools; re-export so the dynamic loader still finds them.
from app.styleos.tools import (  # noqa: E402
    scan_cosmetic,
    match_cosmetics,
    _fetch_cosmetics_by_search,
    _fetch_cosmetics_for_profile,
    _derive_skin_profile,
    get_cosmetics_by_skin_type,
    search_cosmetics_by_skin_type,
    recommend_cosmetics,
    _google_geocode,
    _search_single_location,
    search_nearby_places,
    navigate_app,
    _fetch_all_garments,
    _fetch_outfits,
    _outfit_gender,
    _geocode_location,
    _fetch_weather,
    _has_heavy_outerwear,
    _parse_temp_c,
    get_outfits_by_category,
    recommend_garments,
    search_outfits_by_category,
    generate_outfit_image,
)

# Legal tools live in app.legal.tools; re-export so the dynamic loader still finds them.
from app.legal.tools import (  # noqa: E402
    DOCUMENT_TEMPLATES,
    UK_DOCUMENT_TEMPLATES,
    _CASE_DOCUMENT_CHAR_CAP,
    _juris_call,
    _normalize_search_results,
    search_jurisprudence,
    search_republic_acts,
    _merge_ra_results,
    get_case,
    get_republic_act,
    get_case_document,
    _uk_tool_failure,
    _uk_call,
    _uk_direct_fallback_enabled,
    _uk_with_direct_fallback,
    _uk_result_rows,
    case_law_search,
    judgment_get_header,
    judgment_get_index,
    judgment_get_paragraph,
    case_law_grep_judgment,
    legislation_search,
    _uk_recommendation_rows,
    get_legal_recommendation_uk,
    legislation_get_toc,
    legislation_get_section,
    citations_parse,
    citations_resolve,
    citations_network,
    citations_format_oscola,
    parliament_search_hansard,
    parliament_policy_position_summary,
    parliament_find_member,
    parliament_member_debates,
    parliament_member_interests,
    parliament_search_petitions,
    parliament_get_debate_divisions,
    parliament_get_debate_contributions,
    parliament_lookup_by_column,
    bills_search_bills,
    bills_get_bill,
    votes_search_divisions,
    votes_get_division,
    committees_search_committees,
    committees_get_committee,
    committees_search_evidence,
    hmrc_get_vat_rate,
    hmrc_check_mtd_status,
    hmrc_search_guidance,
    uk_legal_mcp_list_prompts,
    uk_legal_mcp_get_prompt,
    uk_legal_mcp_list_resources,
    uk_legal_mcp_read_resource,
    _kr_call,
    search_law,
    get_law_text,
    get_annexes,
    ordinance_radar,
    legal_research,
    legal_analysis,
    search_decisions,
    get_decision_text,
    discover_tools_kr,
    execute_tool_kr,
    generate_legal_document_uk,
    draft_pleading_uk,
    get_legal_recommendation,
    analyze_document,
    _uk_extract_document_text,
    analyze_document_uk,
    generate_legal_document,
    _format_authorities_block,
    draft_pleading,
    _http_get_json,
    _http_post_json,
    _http_patch_json,
    find_lawyer,
    mark_case_ready,
    create_case,
)

