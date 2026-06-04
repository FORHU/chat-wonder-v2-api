import os
import io
import json
import time
import hashlib
import tempfile
import logging
import urllib.request
import urllib.parse
from datetime import datetime, date

# Simple in-memory cache with TTL
_search_cache = {}
_logger = logging.getLogger(__name__)
CACHE_TTL_SECONDS = 30

# Garments API
_GARMENTS_API_BASE = os.getenv("GARMENTS_API_BASE", "http://ec2-52-77-250-122.ap-southeast-1.compute.amazonaws.com:3007/api/external/garments")
_GARMENTS_CACHE_TTL = 300  # 5 minutes
_garments_cache: dict = {"data": None, "timestamp": 0}

# Outfits API (pre-composed outfits)
_OUTFITS_API_BASE = os.getenv("OUTFITS_API_BASE", "http://ec2-52-77-250-122.ap-southeast-1.compute.amazonaws.com:3007/api/external/outfits")
_OUTFITS_CACHE_TTL = 300  # 5 minutes
_outfits_cache: dict = {"data": None, "timestamp": 0}

# Cosmetics API
_COSMETICS_API_BASE = os.getenv("COSMETICS_API_BASE", "http://ec2-52-77-250-122.ap-southeast-1.compute.amazonaws.com:3007/api/external/cosmetics")
_COSMETICS_CACHE_TTL = 300  # 5 minutes
_cosmetics_cache: dict = {}   # keyed by search term
_DEFAULT_LAT = 14.5995  # Manila
_DEFAULT_LON = 120.9842

_WMO_DESCRIPTIONS = {
    0: ("clear sky", "sunny"), 1: ("mainly clear", "mostly sunny"),
    2: ("partly cloudy", "partly cloudy"), 3: ("overcast", "cloudy"),
    45: ("foggy", "foggy"), 48: ("icy fog", "foggy"),
    51: ("light drizzle", "drizzly"), 53: ("moderate drizzle", "drizzly"), 55: ("heavy drizzle", "drizzly"),
    61: ("light rain", "rainy"), 63: ("moderate rain", "rainy"), 65: ("heavy rain", "rainy"),
    71: ("light snow", "snowy"), 73: ("moderate snow", "snowy"), 75: ("heavy snow", "snowy"),
    80: ("rain showers", "showery"), 81: ("moderate showers", "showery"), 82: ("heavy showers", "showery"),
    95: ("thunderstorm", "stormy"), 96: ("thunderstorm with hail", "stormy"), 99: ("severe thunderstorm", "stormy"),
}
_RAINY_CODES = {51, 53, 55, 61, 63, 65, 80, 81, 82, 95, 96, 99}


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


def _fetch_full_case_content(item_id: int, doc_type: str = None) -> dict:
    import psycopg2
    from psycopg2.extras import RealDictCursor

    try:
        database_url = os.getenv("LEGAL_DATABASE_URL")
        if not database_url:
            return {"success": False, "error": "LEGAL_DATABASE_URL not configured"}

        conn = psycopg2.connect(database_url)
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute(
            """
            SELECT id, source_url AS url, title, category, bucket_slug, year,
                   full_text AS content, summary, concise_summary, metadata_json
            FROM documents WHERE id = %s
            """,
            (item_id,),
        )
        result = cursor.fetchone()
        cursor.close()
        conn.close()

        if not result:
            return {"success": False, "error": f"Document {item_id} not found"}

        doc = dict(result)
        metadata = doc.get("metadata_json") or {}
        doc["gr_number"] = metadata.get("gr_number", "")
        doc["law_number"] = metadata.get("law_number", "")
        doc["case_number"] = metadata.get("case_number", "")
        return {"success": True, "document": doc}

    except Exception as e:
        return {"success": False, "error": str(e)}


def search_legal(query: str, page: int = 1, limit: int = 5, content_types: list = None) -> dict:
    """Search for Philippine law content (cases, statutes, SC E-Library documents)."""
    offset = (page - 1) * limit
    content_types_str = ",".join(sorted(content_types)) if content_types else "all"
    cache_key = hashlib.md5(f"legal:{query}:{page}:{limit}:{content_types_str}".encode()).hexdigest()
    if cache_key in _search_cache:
        cached = _search_cache[cache_key]
        if time.time() - cached["timestamp"] < CACHE_TTL_SECONDS:
            cached["result"]["cached"] = True
            _logger.info("search_legal cache HIT query=%r", query[:80])
            return cached["result"]
        del _search_cache[cache_key]

    try:
        from legal_rag.router import legal_search as legal_rag_search

        t0 = time.perf_counter()
        category = content_types[0] if content_types else None
        rag_payload = legal_rag_search(query=query.strip(), limit=max(limit * page, limit), category=category)
        # If category filter returns no results, retry without it
        if category and (not rag_payload or not rag_payload.get("results")):
            _logger.info("search_legal category=%r returned 0 results, retrying unfiltered", category)
            rag_payload = legal_rag_search(query=query.strip(), limit=max(limit * page, limit))
        _logger.info("search_legal MISS query=%r total=%.0fms", query[:80], (time.perf_counter() - t0) * 1000)
        rag_results = rag_payload.get("results", []) if isinstance(rag_payload, dict) else []

        mapped = []
        for row in rag_results:
            mapped.append({
                "id": str(row.get("id") or ""),
                "item_id": str(row.get("id") or ""),
                "score": float(row.get("final_score", 0.0) or 0.0),
                "type": row.get("category", "legal_document"),
                "title": row.get("title"),
                "url": row.get("source_url"),
                "text": row.get("full_text") or row.get("summary") or row.get("snippet", ""),
                "snippet": row.get("snippet", ""),
                "metadata": {
                    "category": row.get("category"),
                    "bucket_slug": row.get("bucket_slug"),
                    "year": row.get("year"),
                    "source_url": row.get("source_url"),
                    "s3_json_path": row.get("s3_json_path"),
                },
            })

        paged = mapped[offset: offset + limit]
        result = {
            "success": True,
            "query": query,
            "page": page,
            "limit": limit,
            "total_results": len(mapped),
            "results": paged,
            "search_type": "hybrid_rag",
            "cached": False,
        }
        _search_cache[cache_key] = {"timestamp": time.time(), "result": result.copy()}
        return result
    except Exception as e:
        return {"success": False, "error": str(e), "message": f"Legal search failed: {str(e)}"}


def summarize_legal_case(case_identifier: str, summary_type: str = "brief") -> dict:
    """Generate a structured summary of a Philippine legal case."""
    from openai import OpenAI

    search_result = search_legal(case_identifier, page=1, limit=1, content_types=["case"])
    if not search_result.get("success"):
        return {"success": False, "error": search_result.get("error", "Search failed")}

    cases = search_result.get("results", [])
    if not cases:
        search_result = search_legal(case_identifier, page=1, limit=1)
        all_docs = search_result.get("results", [])
        if not all_docs:
            return {"success": False, "error": f"No legal documents found matching '{case_identifier}'"}
        cases = [all_docs[0]]

    case = cases[0]

    prompts = {
        "brief": "Provide a concise 2-3 paragraph summary covering: key facts, main legal issue(s), and the Supreme Court's ruling.",
        "detailed": (
            "Provide a structured legal summary with sections: CASE INFORMATION, FACTS, ISSUES, RULING, DOCTRINE, DISPOSITIVE PORTION."
        ),
        "legal-analysis": (
            "Provide a comprehensive legal analysis with: CASE INFORMATION, PROCEDURAL HISTORY, STATEMENT OF FACTS, "
            "ISSUES PRESENTED, RULING AND RATIO DECIDENDI, OBITER DICTA, DOCTRINAL SIGNIFICANCE, RELATED CASES, PRACTICAL APPLICATION."
        ),
    }
    if summary_type not in prompts:
        summary_type = "brief"

    item_id = case.get("item_id")
    case_text = case.get("text", "")
    if item_id:
        full_result = _fetch_full_case_content(int(item_id))
        if full_result.get("success"):
            full_doc = full_result["document"]
            case_text = full_doc.get("content") or case_text
            case.update(full_doc)

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    case_content = (
        f"Case Title: {case.get('title', 'Unknown')}\n"
        f"GR Number: {case.get('gr_number', case.get('case_number', 'N/A'))}\n"
        f"Year: {case.get('year', 'N/A')}\n"
        f"URL: {case.get('url') or case.get('source_url', '')}\n\n"
        f"Full Content:\n{case_text}"
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are a Philippine legal research assistant specializing in Supreme Court jurisprudence.",
                },
                {"role": "user", "content": f"{prompts[summary_type]}\n\n{case_content}"},
            ],
            temperature=0.3,
        )
        return {
            "success": True,
            "case_title": case.get("title", "Unknown"),
            "gr_number": case.get("gr_number", ""),
            "year": case.get("year", ""),
            "url": case.get("url") or case.get("source_url", ""),
            "summary_type": summary_type,
            "summary": response.choices[0].message.content,
            "disclaimer": "This summary is generated by AI for informational purposes. Always verify with the original case document.",
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def get_legal_recommendation(legal_issue: str, user_context: str = None) -> dict:
    """Provide legal information and recommendations for a given legal issue."""
    from openai import OpenAI

    search_result = search_legal(legal_issue, limit=3)
    relevant_materials = []
    if search_result.get("success"):
        for doc in search_result.get("results", [])[:3]:
            relevant_materials.append(f"{doc.get('type', 'Document')}: {doc.get('title', '')}")

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    context_section = f"\n\nUser's Situation: {user_context}" if user_context else ""
    materials_section = (
        "\n\nRelevant Legal Materials Found:\n" + "\n".join(relevant_materials)
        if relevant_materials
        else ""
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a Philippine legal information assistant. Provide helpful legal INFORMATION (not advice).\n"
                        "IMPORTANT RULES:\n"
                        "1. Start with: 'This is general legal information, not legal advice.'\n"
                        "2. Structure your response with clear sections\n"
                        "3. Cite relevant Philippine laws and cases when applicable\n"
                        "4. Always recommend consulting a licensed attorney for specific situations\n"
                        "5. End with: 'For your specific situation, please consult a licensed attorney.'"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Legal Issue: {legal_issue}{context_section}{materials_section}\n\n"
                        "Please provide:\n"
                        "1. BRIEF ANSWER - Direct response to the question\n"
                        "2. LEGAL BASIS - Relevant Philippine law or case\n"
                        "3. EXPLANATION - Plain language explanation\n"
                        "4. PRACTICAL STEPS - What the person can do\n"
                        "5. WHEN TO SEEK LEGAL HELP - Signs they need a lawyer"
                    ),
                },
            ],
            temperature=0.3,
        )
        return {
            "success": True,
            "issue": legal_issue,
            "recommendation": response.choices[0].message.content,
            "relevant_materials": relevant_materials,
            "disclaimer": "This is general legal information, not legal advice. For your specific situation, please consult a licensed attorney.",
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def analyze_document(s3_key: str, filename: str = None) -> dict:
    """Download a document from S3 and return a structured Philippine legal analysis."""
    import s3_storage
    from openai import OpenAI

    fname = filename or os.path.basename(s3_key)
    ext = os.path.splitext(fname)[1].lower()

    tmp_path = os.path.join(tempfile.gettempdir(), os.path.basename(s3_key))
    downloaded = s3_storage.download_from_s3(s3_key, tmp_path)
    if not downloaded:
        return {"success": False, "error": "File not found in S3 or S3 is not configured."}

    extracted_text = ""
    try:
        with open(tmp_path, "rb") as f:
            contents = f.read()

        if len(contents) > 20 * 1024 * 1024:
            return {"success": False, "error": "File too large. Maximum allowed size is 20MB."}

        if ext == ".txt":
            for enc in ["utf-8", "cp1252", "latin-1"]:
                try:
                    extracted_text = contents.decode(enc)
                    break
                except UnicodeDecodeError:
                    continue

        elif ext == ".pdf":
            try:
                import PyPDF2
                reader = PyPDF2.PdfReader(io.BytesIO(contents))
                extracted_text = "\n\n".join(
                    p.extract_text().strip() for p in reader.pages if p.extract_text()
                )
            except ImportError:
                try:
                    import pdfplumber
                    with pdfplumber.open(io.BytesIO(contents)) as pdf:
                        extracted_text = "\n\n".join(p.extract_text() or "" for p in pdf.pages).strip()
                except ImportError:
                    return {"success": False, "error": "PDF library not installed. Run: pip install PyPDF2"}
            except Exception as e:
                return {"success": False, "error": f"Failed to parse PDF: {e}"}

        elif ext in (".docx", ".doc"):
            try:
                import docx
                doc = docx.Document(io.BytesIO(contents))
                extracted_text = "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())
            except ImportError:
                return {"success": False, "error": "DOCX library not installed. Run: pip install python-docx"}
            except Exception as e:
                return {"success": False, "error": f"Failed to parse DOCX: {e}"}

        elif ext in (".mp3", ".wav", ".m4a"):
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                return {"success": False, "error": "OpenAI API key required for audio transcription."}
            try:
                client = OpenAI(api_key=api_key)
                with open(tmp_path, "rb") as audio_file:
                    transcription = client.audio.transcriptions.create(model="whisper-1", file=audio_file)
                extracted_text = transcription.text
            except Exception as e:
                return {"success": False, "error": f"Audio transcription failed: {e}"}

        elif ext in (".png", ".jpg", ".jpeg"):
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                return {"success": False, "error": "OpenAI API key required for image OCR."}
            try:
                import base64, mimetypes
                b64 = base64.b64encode(contents).decode("utf-8")
                mime_type = mimetypes.guess_type(fname)[0] or f"image/{ext[1:]}"
                client = OpenAI(api_key=api_key)
                resp = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": [
                        {"type": "text", "text": "Extract all text from this image exactly as written. If no text, describe the image briefly."},
                        {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{b64}"}},
                    ]}],
                    max_tokens=3000,
                )
                extracted_text = resp.choices[0].message.content.strip()
            except Exception as e:
                return {"success": False, "error": f"Image OCR failed: {e}"}

        else:
            return {"success": False, "error": f"Unsupported file type '{ext}'. Supported: PDF, DOCX, TXT, PNG, JPG, MP3, WAV, M4A."}

        extracted_text = extracted_text.strip()
        if not extracted_text:
            return {"success": False, "error": "No text could be extracted from the document."}

        CHAR_LIMIT = 50000
        truncated = len(extracted_text) > CHAR_LIMIT
        if truncated:
            extracted_text = extracted_text[:CHAR_LIMIT]

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return {"success": False, "error": "OpenAI API key not configured."}

        client = OpenAI(api_key=api_key)
        model = os.getenv("CHAT_MODEL", "gpt-4o-mini")

        system_prompt = (
            "You are an expert Philippine legal document analyst with deep knowledge of the Civil Code, "
            "Revised Penal Code, Labor Code, and Supreme Court decisions. "
            "Analyze the document and produce a structured legal analysis with these sections:\n\n"
            "## 1. Document Overview\n"
            "Document type, parties, date, jurisdiction, and legal purpose.\n\n"
            "## 2. Key Legal Issues & Provisions\n"
            "All significant obligations, rights, conditions, and prohibitions with their implications.\n\n"
            "## 3. Relevant Philippine Laws & Jurisprudence\n"
            "Applicable statutes (Civil Code Articles, RA numbers) and Supreme Court decisions (G.R. numbers).\n\n"
            "## 4. Notable Clauses or Concerns\n"
            "Unusual, ambiguous, or risky clauses with explanation of the legal risk.\n\n"
            "## 5. Parties' Rights & Obligations\n"
            "What each party is entitled to and obligated to do.\n\n"
            "## 6. Potential Legal Issues or Disputes\n"
            "Scenarios that could lead to disputes or enforcement problems and how to mitigate them.\n\n"
            "## 7. Recommendations\n"
            "Specific, actionable legal advice and suggested next steps."
        )
        text_for_ai = extracted_text[:25000]
        if truncated:
            text_for_ai += f"\n\n[Note: Document truncated — only the first {CHAR_LIMIT:,} characters were analyzed.]"

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Document: **{fname}**\n\n---\n\n{text_for_ai}"},
            ],
            temperature=0.2,
            max_tokens=4000,
        )
        ai_summary = response.choices[0].message.content.strip()

        file_url = s3_storage.generate_presigned_get(s3_key)

        return {
            "success": True,
            "filename": fname,
            "s3_key": s3_key,
            "file_url": file_url,
            "ai_summary": ai_summary,
            "char_count": len(extracted_text),
            "truncated": truncated,
        }

    except Exception as e:
        logging.error(f"[analyze_document] Unexpected error: {e}")
        return {"success": False, "error": str(e)}
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


def scan_cosmetic(front_s3_key: str, back_s3_key: str, skin_type: str = "general") -> dict:
    """Analyze a cosmetic product by scanning its front and back label images stored in S3."""
    import s3_storage
    import base64
    import mimetypes
    import tempfile
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return {"success": False, "error": "OpenAI API key not configured."}

    model = os.getenv("COSMETICS_MODEL", "gpt-4o-mini")
    cosmetics_bucket = os.getenv("COSMETICS_S3_BUCKET_NAME")
    cosmetics_region = os.getenv("COSMETICS_AWS_REGION")
    front_s3_key = front_s3_key.lstrip("/") if front_s3_key else ""
    back_s3_key = back_s3_key.lstrip("/")
    logging.info(f"[scan_cosmetic] bucket={cosmetics_bucket} region={cosmetics_region} model={model}")

    client = OpenAI(api_key=api_key)

    def _ocr_from_s3(s3_key: str, prompt: str) -> str:
        fname = os.path.basename(s3_key)
        ext = os.path.splitext(fname)[1].lower()
        tmp_path = os.path.join(tempfile.gettempdir(), fname)
        logging.info(f"[scan_cosmetic] downloading s3_key={s3_key} bucket={cosmetics_bucket} region={cosmetics_region}")
        if not s3_storage.download_from_s3(s3_key, tmp_path, bucket_name=cosmetics_bucket, region=cosmetics_region):
            logging.error(f"[scan_cosmetic] S3 download FAILED for key={s3_key}")
            return ""
        logging.info(f"[scan_cosmetic] S3 download OK, running OCR on {fname}")
        try:
            with open(tmp_path, "rb") as f:
                contents = f.read()
            mime_type = mimetypes.guess_type(fname)[0] or f"image/{ext[1:]}"
            b64 = base64.b64encode(contents).decode("utf-8")
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{b64}"}},
                ]}],
                max_tokens=2000,
            )
            result = resp.choices[0].message.content.strip()
            logging.info(f"[scan_cosmetic] OCR OK for {fname}, chars={len(result)}")
            return result
        except Exception as e:
            logging.error(f"[scan_cosmetic] OCR FAILED for {s3_key}: {e}")
            return ""
        finally:
            try:
                os.remove(tmp_path)
            except Exception:
                pass

    front_text = _ocr_from_s3(
        front_s3_key,
        "Extract the product name, brand name, and any label claims (e.g. 'hypoallergenic', 'dermatologist tested', 'paraben-free') from this cosmetic product image. Return plain text only.",
    ) if front_s3_key else ""
    back_text = _ocr_from_s3(
        back_s3_key,
        "Extract the full ingredient list from this cosmetic product label exactly as written. Return only the ingredients, comma-separated, in the same order as on the label.",
    )

    if not back_text:
        return {"success": False, "error": "Could not extract an ingredient list from the back label image."}

    skin_type_normalized = (skin_type or "general").lower().strip()
    if skin_type_normalized not in {"oily", "dry", "sensitive", "combination", "general"}:
        skin_type_normalized = "general"

    system_prompt = (
        "You are a cosmetic ingredient expert and dermatologist. Analyze cosmetic product ingredients and return "
        "a JSON object with this exact structure:\n"
        "{\n"
        '  "product_name": "...",\n'
        '  "brand": "...",\n'
        '  "label_claims": ["..."],\n'
        '  "skin_type": "...",\n'
        '  "ingredients": [\n'
        '    {\n'
        '      "name": "...",\n'
        '      "function": "...",\n'
        '      "safety_concern": "none | low | moderate | high",\n'
        '      "safety_notes": "...",\n'
        '      "skin_type_verdict": "beneficial | neutral | problematic",\n'
        '      "skin_type_notes": "..."\n'
        '    }\n'
        '  ],\n'
        '  "summary": "markdown overview of the product and key findings for the user\'s skin type"\n'
        "}"
    )
    user_prompt = (
        f"Front label:\n{front_text}\n\n"
        f"Ingredients (back label):\n{back_text}\n\n"
        f"Skin type: {skin_type_normalized}\n\n"
        "Analyze every ingredient and return the JSON."
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            response_format={"type": "json_object"},
        )
        result = json.loads(response.choices[0].message.content.strip())
        return {
            "success": True,
            "front_s3_key": front_s3_key,
            "back_s3_key": back_s3_key,
            "front_image_url": s3_storage.generate_presigned_get(front_s3_key, bucket_name=cosmetics_bucket, region=cosmetics_region),
            "back_image_url": s3_storage.generate_presigned_get(back_s3_key, bucket_name=cosmetics_bucket, region=cosmetics_region),
            **result,
        }
    except json.JSONDecodeError as e:
        return {"success": False, "error": f"Failed to parse AI response: {e}"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def match_cosmetics(
    product_a_name: str,
    product_a_ingredients: list,
    product_b_name: str,
    product_b_ingredients: list,
    skin_type: str = "general",
) -> dict:
    """Check whether two cosmetic products are compatible based on their ingredient lists and skin type."""
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return {"success": False, "error": "OpenAI API key not configured."}

    model = os.getenv("COSMETICS_MODEL", "gpt-4o-mini")
    client = OpenAI(api_key=api_key)

    skin_type_normalized = (skin_type or "general").lower().strip()
    if skin_type_normalized not in {"oily", "dry", "sensitive", "combination", "general"}:
        skin_type_normalized = "general"

    def _ingredient_names(ingredients: list) -> str:
        names = []
        for item in ingredients:
            if isinstance(item, dict):
                names.append(item.get("name", ""))
            elif isinstance(item, str):
                names.append(item)
        return ", ".join(n for n in names if n)

    system_prompt = (
        "You are a cosmetic chemist and dermatologist. Assess whether two cosmetic products are safe to use together.\n\n"
        "Return a JSON object with this exact structure:\n"
        "{\n"
        '  "verdict": "safe | caution | avoid",\n'
        '  "verdict_reason": "one-sentence explanation of the overall verdict",\n'
        '  "conflicts": [\n'
        '    {\n'
        '      "ingredient_a": "...",\n'
        '      "ingredient_b": "...",\n'
        '      "issue": "...",\n'
        '      "severity": "low | moderate | high",\n'
        '      "skin_type_notes": "..."\n'
        '    }\n'
        '  ],\n'
        '  "summary": "markdown summary with overall verdict, key conflicts, and usage recommendation (e.g. use product A in the morning, product B at night)"\n'
        "}\n\n"
        "If there are no conflicts, return an empty conflicts array and verdict of safe."
    )

    user_prompt = (
        f"Skin type: {skin_type_normalized}\n\n"
        f"Product A — {product_a_name}:\n{_ingredient_names(product_a_ingredients)}\n\n"
        f"Product B — {product_b_name}:\n{_ingredient_names(product_b_ingredients)}\n\n"
        "Check for ingredient interactions, efficacy conflicts, and skin-type-specific concerns. Return the JSON."
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            response_format={"type": "json_object"},
        )
        result = json.loads(response.choices[0].message.content.strip())
        return {"success": True, "skin_type": skin_type_normalized, **result}
    except json.JSONDecodeError as e:
        return {"success": False, "error": f"Failed to parse AI response: {e}"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def generate_legal_document(document_type: str, details: dict = None, format: str = "markdown", **kwargs) -> dict:
    """Generate a Philippine legal document based on type and provided details."""
    from openai import OpenAI

    if details is None:
        details = {}
    if kwargs:
        details.update(kwargs)

    doc_type_lower = document_type.lower().replace(" ", "_").replace("-", "_")
    if doc_type_lower not in DOCUMENT_TEMPLATES:
        return {
            "success": False,
            "error": f"Unknown document type: '{document_type}'",
            "available_types": list(DOCUMENT_TEMPLATES.keys()),
        }

    template = DOCUMENT_TEMPLATES[doc_type_lower]
    missing = [f for f in template["required_fields"] if not details.get(f)]
    if missing:
        return {
            "success": False,
            "error": "Missing required fields",
            "missing_fields": missing,
            "required_fields": template["required_fields"],
            "optional_fields": template.get("optional_fields", []),
        }

    filled = dict(details)
    for field in template.get("optional_fields", []):
        if field not in filled:
            filled[field] = "N/A"

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    try:
        prompt = template["prompt"].format(**filled)
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a Philippine legal document drafter. Generate properly formatted legal documents "
                        "following Philippine legal conventions. Use proper legal document structure, formal language, "
                        "leave blanks (___) for missing info, and include proper signature/notary blocks."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        return {
            "success": True,
            "document_type": doc_type_lower,
            "document_name": template["name"],
            "format": format,
            "content": response.choices[0].message.content,
            "fields_used": details,
            "disclaimer": "This is a template document generated by AI. Have it reviewed by a licensed attorney before use.",
            "next_steps": [
                "Review all details for accuracy",
                "Fill in any blank fields (marked with ___)",
                "Have a lawyer review the document",
                "If notarization is required, bring valid ID to a notary public",
            ],
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


# ---------------------------------------------------------------------------
# Cosmetics helpers
# ---------------------------------------------------------------------------

_CONCERN_SEARCH_TERMS = {
    "acne-prone":  "acne",
    "anti-aging":  "anti-aging",
    "brightening": "brightening",
    "dark circles": "dark",
}


def _fetch_cosmetics_by_search(search_term: str) -> list:
    """Fetch all pages for a single search term, using a per-term cache."""
    api_key = os.getenv("GARMENTS_API_KEY", "")
    cached = _cosmetics_cache.get(search_term)
    if cached and (time.time() - cached["timestamp"]) < _COSMETICS_CACHE_TTL:
        return cached["data"]
    try:
        encoded = urllib.parse.quote(search_term)
        first = _http_get_json(
            f"{_COSMETICS_API_BASE}?search={encoded}&page=1&limit=100",
            {"x-api-key": api_key},
        )
        if first.get("status") != "success":
            return (cached or {}).get("data", [])
        data = first["data"]
        items = list(data["items"])
        for page in range(2, data["totalPages"] + 1):
            try:
                pd = _http_get_json(
                    f"{_COSMETICS_API_BASE}?search={encoded}&page={page}&limit=100",
                    {"x-api-key": api_key},
                )
                if pd.get("status") == "success":
                    items.extend(pd["data"]["items"])
            except Exception as e:
                logging.warning(f"[cosmetics] search={search_term!r} page {page} failed: {e}")
        _cosmetics_cache[search_term] = {"data": items, "timestamp": time.time()}
        logging.info(f"[cosmetics] search={search_term!r} fetched {len(items)} items")
        return items
    except Exception as e:
        logging.error(f"[cosmetics] search={search_term!r} fetch failed: {e}")
        return (cached or {}).get("data", [])


def _fetch_cosmetics_for_profile(skin_type: str, concerns: list) -> list:
    """Fetch and deduplicate products relevant to the user's skin type and concerns."""
    search_terms = {skin_type} if skin_type != "general" else set()
    for concern in concerns:
        term = _CONCERN_SEARCH_TERMS.get(concern)
        if term:
            search_terms.add(term)
    if not search_terms:
        search_terms = {"general"}

    seen_ids: set = set()
    combined: list = []
    for term in sorted(search_terms):
        for item in _fetch_cosmetics_by_search(term):
            if item["id"] not in seen_ids:
                seen_ids.add(item["id"])
                combined.append(item)

    logging.info(f"[cosmetics] profile fetch: terms={sorted(search_terms)} total_unique={len(combined)}")
    return combined


def _derive_skin_profile(analysis_output: list) -> tuple:
    """Returns (skin_type, concerns) from skin analysis output array."""
    scores = {item["type"]: item.get("ui_score", item.get("score", 0)) for item in analysis_output}
    oiliness = scores.get("oiliness", 50)
    if oiliness >= 70:
        skin_type = "oily"
    elif oiliness <= 35:
        skin_type = "dry"
    elif 35 < oiliness < 55:
        skin_type = "combination"
    else:
        skin_type = "general"

    concerns = []
    if scores.get("acne", 0) >= 60:
        concerns.append("acne-prone")
    if scores.get("wrinkle", 0) >= 60:
        concerns.append("anti-aging")
    if scores.get("age_spot", 0) >= 60:
        concerns.append("brightening")
    if scores.get("dark_circle_v2", 0) >= 60:
        concerns.append("dark circles")
    return skin_type, concerns


def recommend_cosmetics(skin_analysis_json: str = None, sets: int = 1, weather_json: str = None, location_json: str = None) -> dict:
    """Fetch cosmetics catalogue and return AI-curated skincare routines based on skin analysis scores.

    Each routine covers core slots (CLEANSER, MOISTURIZER) plus optional slots
    (TONER, ESSENCE, EXFOLIANT, SUNSCREEN). Multiple routines have distinct vibes
    targeting different concerns derived from the skin analysis scores, weather, and location.
    """
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return {"success": False, "error": "OpenAI API key not configured."}

    n_sets = max(1, min(int(sets or 1), 3))

    skin_type = "general"
    concerns: list = []
    overall_score = None
    skin_age = None

    if skin_analysis_json:
        try:
            analysis = json.loads(skin_analysis_json)
            output = analysis.get("output", [])
            skin_type, concerns = _derive_skin_profile(output)
            overall_score = analysis.get("overallScore")
            skin_age = analysis.get("skinAge")
        except Exception as e:
            logging.warning(f"[cosmetics] skin_analysis_json parse failed: {e}")

    # Parse climate signals from weather and location
    climate_rules: list[str] = []
    climate_ctx = ""
    if weather_json:
        try:
            weather = json.loads(weather_json)
            humidity = weather.get("humidity")
            uvi = weather.get("uvi")
            temp = weather.get("temp")
            city = weather.get("city") or weather.get("name", "")
            climate_parts = []
            if city:
                climate_parts.append(f"Location: {city}")
            if temp is not None:
                climate_parts.append(f"Temperature: {temp}°C")
            if humidity is not None:
                climate_parts.append(f"Humidity: {humidity}%")
            if uvi is not None:
                climate_parts.append(f"UV index: {uvi}")
            if climate_parts:
                climate_ctx = "; ".join(climate_parts)
            if humidity is not None and float(humidity) > 70:
                climate_rules.append("High humidity: prefer lightweight, gel-based, oil-free formulas; avoid heavy creams")
            if uvi is not None and float(uvi) >= 5:
                climate_rules.append("High UV: ALWAYS include a SUNSCREEN in every routine")
            if temp is not None and float(temp) < 10:
                climate_rules.append("Cold weather: prefer richer cream moisturizers; avoid alcohol-heavy toners")
        except Exception as e:
            logging.warning(f"[cosmetics] weather_json parse failed: {e}")

    if location_json:
        try:
            loc = json.loads(location_json)
            loc_label = loc.get("city") or loc.get("name") or loc.get("label", "")
            is_urban = loc.get("urban", False)
            if is_urban or (loc_label and any(k in loc_label.lower() for k in ("city", "metro", "manila", "jakarta", "bangkok", "singapore", "kuala"))):
                climate_rules.append("Urban/high-pollution environment: prioritise antioxidant ingredients (Vitamin C, niacinamide) for environmental protection")
            if loc_label and not climate_ctx:
                climate_ctx = f"Location: {loc_label}"
        except Exception as e:
            logging.warning(f"[cosmetics] location_json parse failed: {e}")

    filtered = _fetch_cosmetics_for_profile(skin_type, concerns)
    if not filtered:
        return {"success": False, "error": "Cosmetics catalogue is unavailable. Please try again later."}

    cosmetic_lookup = {c["id"]: c for c in filtered}

    # Strip heavy fields before sending to GPT
    _SLIM_KEEP = {"id", "name", "brand", "type", "hexColor", "finish",
                  "oilFree", "hydrating", "spf", "priceAmount", "priceUnit"}
    slim_all = [{k: v for k, v in c.items() if k in _SLIM_KEEP} for c in filtered]

    # Cap at 80 products; sample proportionally by type so all routine slots are covered
    _MAX_PRODUCTS = 80
    if len(slim_all) > _MAX_PRODUCTS:
        from collections import defaultdict
        import random
        by_type: dict = defaultdict(list)
        for p in slim_all:
            by_type[p.get("type", "OTHER")].append(p)
        per_type = max(1, _MAX_PRODUCTS // max(len(by_type), 1))
        slim = []
        for bucket in by_type.values():
            slim.extend(random.sample(bucket, min(per_type, len(bucket))))
        slim = slim[:_MAX_PRODUCTS]
    else:
        slim = slim_all

    concerns_ctx = f"Skin concerns: {', '.join(concerns)}." if concerns else "No specific skin concerns identified."
    score_ctx = ""
    if overall_score is not None:
        score_ctx = f" Overall skin score: {overall_score}/100."
    if skin_age is not None:
        score_ctx += f" Skin age: {skin_age}."

    climate_rules_str = ""
    if climate_rules:
        climate_rules_str = "\nCLIMATE RULES (apply on top of skin rules):\n" + "\n".join(f"- {r}" for r in climate_rules) + "\n"

    system_prompt = (
        f"You are a professional dermatologist and skincare specialist. Curate exactly {n_sets} distinct skincare "
        f"routine(s) from the provided cosmetics catalogue based on the user's skin profile and local climate.\n\n"
        "CRITICAL RULES:\n"
        "1. Each routine must contain EXACTLY ONE product per type slot. Never include two products of the same type in one routine.\n"
        "2. Every routine MUST include CLEANSER and MOISTURIZER (core slots).\n"
        "3. TONER, ESSENCE, EXFOLIANT, SUNSCREEN are optional — include only when beneficial for the concerns.\n"
        "4. Do NOT reuse the same product id across different routines.\n"
        f"5. Each routine must have a DISTINCT vibe that targets a different aspect of the user's skin concerns.\n"
        f"{climate_rules_str}\n"
        "Return a JSON object with this exact structure:\n"
        "{\n"
        '  "sets": [\n'
        '    {\n'
        '      "set_number": 1,\n'
        '      "vibe": "Short vibe name e.g. Acne-Fighting AM Routine",\n'
        '      "concern_note": "One sentence on how this routine addresses the user\'s skin concerns and local climate",\n'
        '      "recommendations": [\n'
        '        {\n'
        '          "id": "...",\n'
        '          "name": "...",\n'
        '          "brand": "...",\n'
        '          "type": "CLEANSER",\n'
        '          "priceAmount": "...",\n'
        '          "priceUnit": "...",\n'
        '          "reason": "Why this product suits the user\'s skin type, concerns, and climate"\n'
        '        }\n'
        '      ]\n'
        '    }\n'
        '  ],\n'
        '  "skin_notes": "Brief markdown overview of what the analysis and climate reveal about the user\'s skin needs",\n'
        '  "routine_tips": ["tip1", "tip2"]\n'
        "}\n\n"
        "Additional rules:\n"
        "- For oily/acne-prone skin: prefer oil-free, non-comedogenic products with salicylic acid or niacinamide\n"
        "- For dry skin: prefer hydrating, cream-based moisturizers; avoid harsh exfoliants\n"
        "- For anti-aging concerns: look for retinol, peptides, vitamin C in the details\n"
        "- For brightening concerns: look for vitamin C, niacinamide, kojic acid\n"
        "- Only use products from the provided catalogue — do not invent items\n"
        f"- Return exactly {n_sets} routine(s) in the sets array"
    )
    climate_prompt = f"\nClimate context: {climate_ctx}" if climate_ctx else ""
    user_prompt = (
        f"Skin type: {skin_type}\n"
        f"{concerns_ctx}{score_ctx}{climate_prompt}\n"
        f"Routines requested: {n_sets}\n\n"
        f"Cosmetics catalogue:\n{json.dumps(slim, ensure_ascii=False)}\n\n"
        f"Curate {n_sets} distinct skincare routine(s) and return the JSON."
    )

    client = OpenAI(api_key=api_key)
    model = os.getenv("CHAT_MODEL", "gpt-4o-mini")
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.5,
            response_format={"type": "json_object"},
        )
        result = json.loads(resp.choices[0].message.content.strip())

        for s in result.get("sets", []):
            seen_types: set = set()
            deduped = []
            for rec in s.get("recommendations", []):
                slot = rec.get("type", "")
                if slot and slot in seen_types:
                    continue
                if slot:
                    seen_types.add(slot)
                full = cosmetic_lookup.get(rec.get("id"), {})
                rec["imageUrl"] = full.get("imageUrl", "")
                deduped.append(rec)
            s["recommendations"] = deduped

        return {
            "success": True,
            "skin_type": skin_type,
            "concerns": concerns,
            "overall_score": overall_score,
            "skin_age": skin_age,
            "sets_requested": n_sets,
            **result,
        }
    except json.JSONDecodeError as e:
        return {"success": False, "error": f"Failed to parse AI response: {e}"}
    except Exception as e:
        return {"success": False, "error": str(e)}


# ---------------------------------------------------------------------------
# Google Places helpers
# ---------------------------------------------------------------------------

_GOOGLE_PLACES_API_BASE = "https://maps.googleapis.com/maps/api/place"
_GOOGLE_GEOCODING_API_BASE = "https://maps.googleapis.com/maps/api/geocode"
_NEARBY_INTENT_KEYWORDS = {"near", "nearby", "around", "close", "closest", "nearest"}


def _google_geocode(location: str, api_key: str) -> tuple:
    """Geocode a place name using Google Geocoding API. Returns (lat, lng)."""
    try:
        url = (
            f"{_GOOGLE_GEOCODING_API_BASE}/json"
            f"?address={urllib.parse.quote(location)}"
            f"&region=PH&components=country:PH"
            f"&key={api_key}"
        )
        data = _http_get_json(url)
        results = data.get("results", [])
        if results:
            loc = results[0]["geometry"]["location"]
            return (loc["lat"], loc["lng"])
    except Exception as e:
        logging.warning(f"[google_geocode] failed for '{location}': {e}")
    return (_DEFAULT_LAT, _DEFAULT_LON)


def _search_single_location(api_key: str, lat: float, lng: float, query: str, radius: int, location_label: str) -> dict:
    """Execute one Google Places search for a single lat/lng and return a result dict."""
    query_lower = query.lower()
    use_nearby = any(kw in query_lower for kw in _NEARBY_INTENT_KEYWORDS) or len(query.split()) <= 4

    if use_nearby:
        url = (
            f"{_GOOGLE_PLACES_API_BASE}/nearbysearch/json"
            f"?location={lat},{lng}"
            f"&radius={radius}"
            f"&keyword={urllib.parse.quote(query)}"
            f"&key={api_key}"
        )
    else:
        url = (
            f"{_GOOGLE_PLACES_API_BASE}/textsearch/json"
            f"?query={urllib.parse.quote(query)}"
            f"&location={lat},{lng}"
            f"&radius={radius}"
            f"&key={api_key}"
        )

    data = _http_get_json(url)
    status = data.get("status")
    if status not in ("OK", "ZERO_RESULTS"):
        return {"success": False, "location_label": location_label, "error": f"Google Places API error: {status}"}

    places = []
    for p in data.get("results", [])[:20]:
        loc = p.get("geometry", {}).get("location", {})
        photos = p.get("photos", [])
        opening = p.get("opening_hours", {})
        photo_ref = photos[0].get("photo_reference", "") if photos else ""
        photo_url = (
            f"{_GOOGLE_PLACES_API_BASE}/photo?maxwidth=400"
            f"&photo_reference={photo_ref}&key={api_key}"
        ) if photo_ref else ""
        places.append({
            "name": p.get("name", ""),
            "address": p.get("vicinity") or p.get("formatted_address", ""),
            "rating": p.get("rating"),
            "user_ratings_total": p.get("user_ratings_total"),
            "place_id": p.get("place_id", ""),
            "types": p.get("types", []),
            "lat": loc.get("lat"),
            "lng": loc.get("lng"),
            "open_now": opening.get("open_now"),
            "photo_url": photo_url,
            "price_level": p.get("price_level"),
            "phone_number": None,
            "website": None,
        })

    return {
        "success": True,
        "query": query,
        "location_label": location_label,
        "lat": lat,
        "lng": lng,
        "radius": radius,
        "search_mode": "nearby" if use_nearby else "text",
        "total_results": len(places),
        "places": places,
    }


def search_nearby_places(
    query: str,
    lat: float = 0.0,
    lng: float = 0.0,
    radius: int = 1500,
    location_name: str = None,
) -> dict:
    """Search for nearby places using Google Places API (Nearby Search or Text Search).

    location_name can be a single place name (e.g. "SM Baguio") or a comma-separated
    list of names (e.g. "SM Baguio, La Union, Vigan City") for multi-location queries.
    When provided, GPS lat/lng is ignored and each name is geocoded automatically.
    """
    api_key = os.getenv("GOOGLE_PLACES_API_KEY", "")
    if not api_key:
        return {"success": False, "error": "GOOGLE_PLACES_API_KEY not configured."}

    try:
        # Multi-location: comma-separated names → search each and return array
        if location_name and "," in location_name:
            names = [n.strip() for n in location_name.split(",") if n.strip()]
            results = []
            for name in names:
                try:
                    glat, glng = _google_geocode(name, api_key)
                    logging.info(f"[search_nearby_places] geocoded '{name}' -> ({glat}, {glng})")
                    result = _search_single_location(api_key, glat, glng, query, radius, name)
                except Exception as e:
                    result = {"success": False, "location_label": name, "error": str(e)}
                results.append(result)
            total = sum(r.get("total_results", 0) for r in results if r.get("success"))
            return {
                "success": True,
                "query": query,
                "multi_location": True,
                "total_results": total,
                "results": results,
            }

        # Single named destination: embed location in the text search query directly.
        # This avoids geocoding ambiguity (e.g. "La Union" geocoding to a barangay
        # instead of La Union Province).
        if location_name:
            full_query = f"{query} in {location_name}"
            url = (
                f"{_GOOGLE_PLACES_API_BASE}/textsearch/json"
                f"?query={urllib.parse.quote(full_query)}"
                f"&key={api_key}"
            )
            logging.info(f"[search_nearby_places] text search '{full_query}'")
            data = _http_get_json(url)
            status = data.get("status")
            if status not in ("OK", "ZERO_RESULTS"):
                return {"success": False, "location_label": location_name, "error": f"Google Places API error: {status}"}
            places = []
            for p in data.get("results", [])[:20]:
                loc = p.get("geometry", {}).get("location", {})
                photos = p.get("photos", [])
                opening = p.get("opening_hours", {})
                photo_ref = photos[0].get("photo_reference", "") if photos else ""
                photo_url = (
                    f"{_GOOGLE_PLACES_API_BASE}/photo?maxwidth=400"
                    f"&photo_reference={photo_ref}&key={api_key}"
                ) if photo_ref else ""
                places.append({
                    "name": p.get("name", ""),
                    "address": p.get("vicinity") or p.get("formatted_address", ""),
                    "rating": p.get("rating"),
                    "user_ratings_total": p.get("user_ratings_total"),
                    "place_id": p.get("place_id", ""),
                    "types": p.get("types", []),
                    "lat": loc.get("lat"),
                    "lng": loc.get("lng"),
                    "open_now": opening.get("open_now"),
                    "photo_url": photo_url,
                    "price_level": p.get("price_level"),
                    "phone_number": None,
                    "website": None,
                })
            return {
                "success": True,
                "query": full_query,
                "location_label": location_name,
                "search_mode": "text",
                "total_results": len(places),
                "places": places,
            }

        location_label = f"{lat:.4f},{lng:.4f}"
        return _search_single_location(api_key, lat, lng, query, radius, location_label)

    except Exception as e:
        logging.error(f"[search_nearby_places] failed: {e}")
        return {"success": False, "error": str(e)}


# ---------------------------------------------------------------------------
# Garments helpers
# ---------------------------------------------------------------------------

def _http_get_json(url: str, headers: dict = None) -> dict:
    req = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _fetch_all_garments() -> list:
    api_key = os.getenv("GARMENTS_API_KEY", "")
    if _garments_cache["data"] and (time.time() - _garments_cache["timestamp"]) < _GARMENTS_CACHE_TTL:
        return _garments_cache["data"]
    try:
        first = _http_get_json(f"{_GARMENTS_API_BASE}?page=1&limit=100", {"x-api-key": api_key})
        if first.get("status") != "success":
            return _garments_cache["data"] or []
        data = first["data"]
        items = list(data["items"])
        for page in range(2, data["totalPages"] + 1):
            try:
                pd = _http_get_json(f"{_GARMENTS_API_BASE}?page={page}&limit=100", {"x-api-key": api_key})
                if pd.get("status") == "success":
                    items.extend(pd["data"]["items"])
            except Exception as e:
                logging.warning(f"[garments] page {page} fetch failed: {e}")
        _garments_cache["data"] = items
        _garments_cache["timestamp"] = time.time()
        return items
    except Exception as e:
        logging.error(f"[garments] catalogue fetch failed: {e}")
        return _garments_cache["data"] or []


def _fetch_all_outfits() -> list:
    api_key = os.getenv("GARMENTS_API_KEY", "")
    if _outfits_cache["data"] and (time.time() - _outfits_cache["timestamp"]) < _OUTFITS_CACHE_TTL:
        return _outfits_cache["data"]
    try:
        first = _http_get_json(f"{_OUTFITS_API_BASE}?page=1&limit=100", {"x-api-key": api_key})
        if first.get("status") != "success":
            return _outfits_cache["data"] or []
        data = first["data"]
        items = list(data["items"])
        for page in range(2, data["totalPages"] + 1):
            try:
                pd = _http_get_json(f"{_OUTFITS_API_BASE}?page={page}&limit=100", {"x-api-key": api_key})
                if pd.get("status") == "success":
                    items.extend(pd["data"]["items"])
            except Exception as e:
                logging.warning(f"[outfits] page {page} fetch failed: {e}")
        _outfits_cache["data"] = items
        _outfits_cache["timestamp"] = time.time()
        return items
    except Exception as e:
        logging.error(f"[outfits] catalogue fetch failed: {e}")
        return _outfits_cache["data"] or []


def _outfit_gender(outfit: dict) -> str:
    """Derive outfit gender from its garments. Returns MALE, FEMALE, or UNISEX."""
    genders = {g["garment"]["gender"] for g in outfit.get("items", []) if g.get("garment")}
    if genders == {"MALE"}:
        return "MALE"
    if genders == {"FEMALE"}:
        return "FEMALE"
    return "UNISEX"


def _geocode_location(location: str) -> tuple:
    if not location:
        return (_DEFAULT_LAT, _DEFAULT_LON)
    parts = location.replace(" ", "").split(",")
    if len(parts) == 2:
        try:
            return (float(parts[0]), float(parts[1]))
        except ValueError:
            pass
    try:
        data = _http_get_json(
            f"https://geocoding-api.open-meteo.com/v1/search?name={urllib.parse.quote(location)}&count=1&format=json"
        )
        results = data.get("results", [])
        if results:
            return (results[0]["latitude"], results[0]["longitude"])
    except Exception as e:
        logging.warning(f"[garments] geocode failed for '{location}': {e}")
    return (_DEFAULT_LAT, _DEFAULT_LON)


def _fetch_weather(lat: float, lon: float, event_date_str: str) -> dict:
    today = date.today()
    try:
        event_date = datetime.strptime(event_date_str, "%Y-%m-%d").date()
    except ValueError:
        event_date = today
    days_ahead = (event_date - today).days
    if days_ahead > 16 or days_ahead < 0:
        from calendar import month_name as _mn
        return {"estimated": True, "date": event_date_str, "month": event_date.month, "month_name": _mn[event_date.month]}
    try:
        if days_ahead == 0:
            url = (
                f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
                "&current=temperature_2m,weathercode,precipitation,relative_humidity_2m&timezone=auto"
            )
            d = _http_get_json(url).get("current", {})
            wmo = d.get("weathercode", 0)
            temp = d.get("temperature_2m")
            precip = d.get("precipitation", 0)
        else:
            url = (
                f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
                f"&daily=weathercode,temperature_2m_max,temperature_2m_min,precipitation_sum"
                f"&timezone=auto&start_date={event_date_str}&end_date={event_date_str}"
            )
            daily = _http_get_json(url).get("daily", {})
            wmo = (daily.get("weathercode") or [0])[0]
            tmax = (daily.get("temperature_2m_max") or [None])[0]
            tmin = (daily.get("temperature_2m_min") or [None])[0]
            temp = round(((tmax or 0) + (tmin or 0)) / 2, 1) if tmax and tmin else None
            precip = (daily.get("precipitation_sum") or [0])[0]
        desc, cond = _WMO_DESCRIPTIONS.get(wmo, ("unknown conditions", "unknown"))
        return {
            "estimated": False, "date": event_date_str,
            "temperature_c": temp, "weathercode": wmo,
            "description": desc, "condition": cond,
            "precipitation_mm": precip or 0,
            "is_rainy": wmo in _RAINY_CODES or (precip or 0) > 1,
            "is_hot": (temp or 0) >= 32,
            "is_cold": (temp or 99) <= 20,
        }
    except Exception as e:
        logging.error(f"[garments] weather fetch failed: {e}")
        return {"estimated": True, "date": event_date_str, "error": str(e)}


# ---------------------------------------------------------------------------
# Garment recommendation function
# ---------------------------------------------------------------------------

def recommend_garments(
    gender: str,
    event_type: str = None,
    event_date: str = None,
    location: str = None,
    sets: int = 4,
    weather_json: str = None,
) -> dict:
    """Fetch live weather and pre-composed outfit catalogue, then return AI-selected outfit sets as JSON.

    AI selects the best pre-composed outfits from the external catalogue based on weather,
    event type, gender, and local fashion trends. Each set maps to one complete outfit with
    an outfit-level hero image and individual garment breakdown.
    """
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return {"success": False, "error": "OpenAI API key not configured."}

    gender_upper = (gender or "").strip().upper()
    if gender_upper not in ("MALE", "FEMALE"):
        return {"success": False, "error": "gender must be 'MALE' or 'FEMALE'."}

    n_sets = max(1, min(int(sets or 1), 4))

    resolved_date = date.today().isoformat()
    if event_date:
        try:
            datetime.strptime(event_date, "%Y-%m-%d")
            resolved_date = event_date
        except ValueError:
            pass

    frontend_weather: dict = {}
    frontend_lat, frontend_lon = None, None
    if weather_json:
        try:
            frontend_weather = json.loads(weather_json)
            frontend_weather.setdefault("estimated", False)
            frontend_lat = frontend_weather.pop("lat", None)
            frontend_lon = frontend_weather.pop("lon", None)
        except Exception as e:
            logging.warning(f"[outfits] weather_json parse failed: {e}")

    if location:
        location_label = location
        lat, lon = _geocode_location(location)
        weather = _fetch_weather(lat, lon, resolved_date)
    elif frontend_lat is not None and frontend_lon is not None:
        location_label = "your current location"
        lat, lon = float(frontend_lat), float(frontend_lon)
        weather = frontend_weather or _fetch_weather(lat, lon, resolved_date)
    else:
        location_label = "Manila, Philippines"
        lat, lon = _DEFAULT_LAT, _DEFAULT_LON
        weather = _fetch_weather(lat, lon, resolved_date)

    all_outfits = _fetch_all_outfits()
    if not all_outfits:
        return {"success": False, "error": "Outfit catalogue is unavailable. Please try again later."}

    filtered = [o for o in all_outfits if _outfit_gender(o) in (gender_upper, "UNISEX")]
    if not filtered:
        filtered = all_outfits

    outfit_lookup = {o["id"]: o for o in filtered}

    # Slim catalogue for GPT — no imageUrls to save tokens
    slim = [
        {
            "id": o["id"],
            "name": o["name"],
            "description": o["description"],
            "garments": [
                {
                    "name": g["garment"]["name"],
                    "garmentType": g["garment"]["garmentType"],
                    "fittingSlot": g["garment"]["fittingSlot"],
                    "category": g["garment"]["category"],
                    "layerLevel": g["garment"]["layerLevel"],
                    "silhouette": g["garment"].get("silhouette", ""),
                }
                for g in o.get("items", [])
                if g.get("garment")
            ],
        }
        for o in filtered
    ]

    if weather.get("estimated"):
        month_label = weather.get("month_name", "this time of year")
        weather_ctx = (
            f"No live forecast available for {resolved_date}. "
            f"Use your knowledge of typical {month_label} climate at {location_label} to guide recommendations."
        )
    else:
        temp = weather.get("temperature_c")
        temp_str = f"{temp:.1f}°C" if temp is not None else "unknown temperature"
        weather_ctx = f"Weather on {resolved_date} at {location_label}: {weather['description']}, {temp_str}, precipitation {weather['precipitation_mm']}mm."
        if weather.get("is_rainy"):
            weather_ctx += " Rain expected — favor water-resistant or quick-dry outfits."
        elif weather.get("is_hot"):
            weather_ctx += " Hot — favor breathable, lightweight outfits in lighter colors."
        elif weather.get("is_cold"):
            weather_ctx += " Cool — favor layered or warmer outfits."

    event_ctx = f"Event/occasion: {event_type}." if event_type else "No specific event — recommend versatile everyday outfits."

    system_prompt = (
        f"You are a personal stylist. Select exactly {n_sets} distinct pre-composed outfits from the provided catalogue "
        f"based on weather, event, gender, and local fashion trends at the user's location.\n\n"
        "CRITICAL RULES:\n"
        f"1. Select EXACTLY {n_sets} distinct outfits. Do NOT select the same outfit twice.\n"
        "2. Each selected outfit must have a distinct vibe reflecting different local fashion trends.\n"
        "3. Only select outfit IDs from the provided catalogue — do not invent IDs.\n"
        f"4. Each set must reflect a different sub-culture or style direction at {location_label}.\n\n"
        "Return a JSON object with this exact structure:\n"
        "{\n"
        '  "sets": [\n'
        '    {\n'
        '      "set_number": 1,\n'
        '      "outfit_id": "exact id from catalogue",\n'
        '      "vibe": "Short vibe name e.g. BGC Smart Casual",\n'
        '      "trend_note": "One sentence on how this vibe reflects local fashion culture at the location",\n'
        '      "reason": "Why this outfit suits the weather and event"\n'
        '    }\n'
        '  ],\n'
        '  "weather_note": "How weather conditions shaped these picks",\n'
        '  "styling_tips": ["tip1", "tip2"]\n'
        "}\n\n"
        "Additional rules:\n"
        "- Rain: prefer outfits with darker colors, avoid suede items\n"
        "- Heat: prefer outfits with breathable fabrics and lighter colors\n"
        "- Cold: prefer layered outfits\n"
        "- Formal events: prefer outfits with Formal/Business category garments\n"
        "- Casual events: prefer outfits with Casual/SmartCasual category garments\n"
        f"- Return exactly {n_sets} set(s) in the sets array"
    )
    user_prompt = (
        f"Gender: {gender_upper}\n"
        f"Requested outfits: {n_sets}\n"
        f"{weather_ctx}\n"
        f"{event_ctx}\n\n"
        f"Outfit catalogue:\n{json.dumps(slim, ensure_ascii=False)}\n\n"
        f"Select {n_sets} distinct outfits and return the JSON."
    )

    client = OpenAI(api_key=api_key)
    model = os.getenv("CHAT_MODEL", "gpt-4o-mini")
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.5,
            response_format={"type": "json_object"},
        )
        result = json.loads(resp.choices[0].message.content.strip())

        # Hydrate each set with full outfit data and garment breakdown
        for s in result.get("sets", []):
            outfit = outfit_lookup.get(s.get("outfit_id"), {})
            s["outfit_name"] = outfit.get("name", "")
            s["outfit_description"] = outfit.get("description", "")
            s["outfit_imageUrl"] = outfit.get("imageUrl", "")
            s["recommendations"] = [
                {
                    "id": g["garment"]["id"],
                    "name": g["garment"]["name"],
                    "description": g["garment"]["description"],
                    "imageUrl": g["garment"]["imageUrl"],
                    "fittingSlot": g["garment"]["fittingSlot"],
                    "garmentType": g["garment"]["garmentType"],
                    "category": g["garment"]["category"],
                    "layerLevel": g["garment"]["layerLevel"],
                    "silhouette": g["garment"].get("silhouette", ""),
                }
                for g in outfit.get("items", [])
                if g.get("garment")
            ]

        return {
            "success": True,
            "gender": gender_upper,
            "event_type": event_type,
            "event_date": resolved_date,
            "location": location_label,
            "coordinates": {"lat": lat, "lon": lon},
            "weather": weather,
            "sets_requested": n_sets,
            **result,
        }
    except json.JSONDecodeError as e:
        return {"success": False, "error": f"Failed to parse AI response: {e}"}
    except Exception as e:
        return {"success": False, "error": str(e)}
