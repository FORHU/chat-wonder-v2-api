# coding: utf-8

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

def chat_via_streaming(prompt, auto_approve=False):
    print(f"User Input: {prompt}")
    first = True
    last_char = ""
    for chunk in call_chat_stream(prompt, session_id, auto_approve=auto_approve):
        if not chunk:
            continue
        if first:
            print()  # blank line before response starts
            first = False
        print(chunk, end="", flush=True)
        last_char = chunk[-1]
    if last_char and last_char != chr(10):
        print()
    print()
    _continue()

# _continue("quit")
