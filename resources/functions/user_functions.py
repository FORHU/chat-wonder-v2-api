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



DOCUMENT_TEMPLATES = {
    "affidavit_of_loss": {
        "name": "Affidavit of Loss",
        "required_fields": ["affiant_name", "item_lost", "date_lost", "circumstances"],
        "optional_fields": ["affiant_address", "purpose"],
        "prompt": """Generate a Philippine-format Affidavit of Loss with:
- Affiant: {affiant_name}
- Address: {affiant_address}
- Lost item: {item_lost}
- Date of loss: {date_lost}
- Circumstances: {circumstances}
- Purpose: {purpose}

Include:
1. Republic of the Philippines header
2. City/Municipality for venue (use "______, Philippines" if not specified)
3. S.S. notation
4. Proper affidavit structure with numbered paragraphs
5. Affiant's statement under oath
6. Jurat section for notarization (leave blanks for notary details)
7. Space for affiant's signature and government ID

Format in proper legal document style."""
    },
    "demand_letter": {
        "name": "Demand Letter",
        "required_fields": ["sender_name", "recipient_name", "amount", "reason", "deadline_days"],
        "optional_fields": ["sender_address", "recipient_address", "reference_number"],
        "prompt": """Generate a Philippine-format Demand Letter with:
- Sender: {sender_name}
- Sender Address: {sender_address}
- Recipient: {recipient_name}
- Recipient Address: {recipient_address}
- Amount demanded: PHP {amount}
- Reason/Cause: {reason}
- Deadline: {deadline_days} days from receipt

Include:
1. Current date
2. Formal salutation
3. Clear statement of demand
4. Factual basis for the claim
5. Specific amount with breakdown if applicable
6. Deadline for compliance
7. Warning of legal action if ignored
8. Formal closing
9. Space for sender's signature

Cite relevant Philippine law if applicable (e.g., Civil Code provisions)."""
    },
    "power_of_attorney": {
        "name": "Special Power of Attorney (SPA)",
        "required_fields": ["principal_name", "attorney_name", "powers_granted"],
        "optional_fields": ["principal_address", "attorney_address", "duration", "property_details"],
        "prompt": """Generate a Philippine-format Special Power of Attorney (SPA) with:
- Principal (Grantor): {principal_name}
- Principal Address: {principal_address}
- Attorney-in-Fact: {attorney_name}
- Attorney Address: {attorney_address}
- Powers Granted: {powers_granted}
- Duration: {duration}
- Property/Subject Matter: {property_details}

Include:
1. "KNOW ALL MEN BY THESE PRESENTS:" header
2. Recitals (WHEREAS clauses)
3. Granting clause with specific enumerated powers
4. Ratification clause
5. Duration/Revocation terms
6. IN WITNESS WHEREOF clause
7. Signature blocks for Principal
8. Acknowledgment section for notarization
9. Space for witnesses (2 witnesses required)"""
    },
    "authorization_letter": {
        "name": "Authorization Letter",
        "required_fields": ["authorizer_name", "authorized_person", "purpose"],
        "optional_fields": ["authorizer_address", "authorized_person_address", "valid_until", "specific_documents"],
        "prompt": """Generate a Philippine-format Authorization Letter with:
- Authorizer: {authorizer_name}
- Authorizer Address: {authorizer_address}
- Authorized Person: {authorized_person}
- Authorized Person Address: {authorized_person_address}
- Purpose: {purpose}
- Specific Documents/Transactions: {specific_documents}
- Valid Until: {valid_until}

Include:
1. Date
2. "To Whom It May Concern" or specific recipient
3. Clear authorization statement
4. Scope of authorization
5. Validity period
6. Signature of authorizer
7. Note about presenting valid IDs"""
    },
    "promissory_note": {
        "name": "Promissory Note",
        "required_fields": ["borrower_name", "lender_name", "principal_amount", "due_date"],
        "optional_fields": ["interest_rate", "payment_schedule", "collateral"],
        "prompt": """Generate a Philippine-format Promissory Note with:
- Borrower/Maker: {borrower_name}
- Lender/Payee: {lender_name}
- Principal Amount: PHP {principal_amount}
- Due Date: {due_date}
- Interest Rate: {interest_rate}
- Payment Schedule: {payment_schedule}
- Collateral: {collateral}

Include:
1. Amount in words and figures
2. Promise to pay statement
3. Interest terms (if applicable)
4. Default provisions
5. Venue for legal action (Philippine courts)
6. Signature of borrower
7. Date and place of execution
8. Space for witnesses"""
    },
    "lease_contract": {
        "name": "Lease Contract",
        "required_fields": ["lessor_name", "lessee_name", "property_address", "monthly_rent", "lease_term"],
        "optional_fields": ["security_deposit", "advance_payment", "utilities_arrangement", "restrictions"],
        "prompt": """Generate a Philippine-format Lease Contract with:
- Lessor (Landlord): {lessor_name}
- Lessee (Tenant): {lessee_name}
- Property Address: {property_address}
- Monthly Rent: PHP {monthly_rent}
- Lease Term: {lease_term}
- Security Deposit: {security_deposit}
- Advance Payment: {advance_payment}
- Utilities: {utilities_arrangement}
- Restrictions: {restrictions}

Include standard Philippine lease provisions:
1. Parties identification
2. Property description
3. Lease term and renewal
4. Rent payment terms
5. Security deposit terms
6. Maintenance responsibilities
7. Prohibited uses
8. Termination conditions
9. Dispute resolution
10. Signatures of both parties and witnesses"""
    }
}



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

