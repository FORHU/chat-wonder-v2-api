# coding: utf-8

import json
from chatwonder import *

def _continue(purpose=None):
    print("")
    if purpose:
        input(f"Press <ENTER> to {purpose}...")
    else:
        input(f"Press <ENTER> to continue...")
    print("")

# Example: Get session ID via WebSocket
session_id = get_session_id()
print("Session ID:", session_id)
_continue()

# Example: Check and configure HITL settings
# print("Current HITL Status:")
# print(call_get_hitl_status())
# _continue()

# Example: Enable HITL (set auto_approval to False)
# Uncomment the following line to enable HITL approval workflow
# print(call_set_hitl(auto_approval=False))
# _continue()

# Example: Upload embedding vector DB
# embeddings_response = call_install_embeddings("embeddings.pkz", session_id)
# print("Embeddings Response:", embeddings_response)
# _continue()

_STRUCTURED_PREFIXES = ("[GARMENT_DATA]", "[COSMETICS_DATA]", "[MAPS_DATA]", "[NAV_DATA]")

def chat_via_streaming(prompt, auto_approve=False, **extra_fields):
    print(f"User Input: {prompt}")
    first = True
    last_char = ""
    for chunk in call_chat_stream(prompt, session_id, auto_approve=auto_approve, **extra_fields):
        if not chunk:
            continue
        # Pretty-print structured data frames separately
        for prefix in _STRUCTURED_PREFIXES:
            if chunk.startswith(prefix):
                raw = chunk[len(prefix):]
                try:
                    parsed = json.loads(raw)
                    print(f"\n{prefix}")
                    print(json.dumps(parsed, indent=2, ensure_ascii=False))
                except Exception:
                    print(f"\n{chunk}")
                last_char = "\n"
                break
        else:
            if first:
                print()
                first = False
            print(chunk, end="", flush=True)
            last_char = chunk[-1] if chunk else last_char
    if last_char and last_char != chr(10):
        print()
    print()
    _continue()


# _continue("quit")
