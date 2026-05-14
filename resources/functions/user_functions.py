import os
import io
import json
import time
import hashlib
import tempfile
import logging

# Simple in-memory cache with TTL
_search_cache = {}
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
            return cached["result"]
        del _search_cache[cache_key]

    try:
        from legal_rag.router import legal_search as legal_rag_search

        rag_payload = legal_rag_search(query=query.strip(), limit=max(limit * page, limit))
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
