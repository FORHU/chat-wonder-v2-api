# coding: utf-8

import requests
import websocket
import json
import os

BASE_URL = "localhost:8000"          # Modify this appropriately

def _normalize_hitl_decision(decision=None, approved=None):
    """Normalize HITL decision into approved, rejected, or skipped_continue."""
    if decision is not None:
        value = str(decision).strip().lower().replace("-", "_").replace("/", "_")
        aliases = {
            "a": "approved",
            "approve": "approved",
            "approved": "approved",
            "yes": "approved",
            "y": "approved",
            "": "approved",
            "r": "rejected",
            "reject": "rejected",
            "rejected": "rejected",
            "no": "rejected",
            "n": "rejected",
            "s": "skipped_continue",
            "skip": "skipped_continue",
            "skipped": "skipped_continue",
            "continue": "skipped_continue",
            "cont": "skipped_continue",
            "skip_continue": "skipped_continue",
            "skip_cont": "skipped_continue",
            "skipped_continue": "skipped_continue",
            "skipped_by_user_and_continued": "skipped_continue",
        }
        if value in aliases:
            return aliases[value]
        raise ValueError(f"Unknown HITL decision: {decision}")
    if approved is True:
        return "approved"
    if approved is False:
        return "rejected"
    return "approved"


def _prompt_hitl_decision(auto_approve=False):
    """Prompt for HITL decision. Returns (decision, comments)."""
    if auto_approve:
        return "approved", None

    while True:
        user_choice = input("Choose action: [a]pprove / [r]eject / [s]kip+continue: ").strip().lower()
        try:
            decision = _normalize_hitl_decision(user_choice)
            break
        except ValueError:
            print("Please enter a, r, or s.")

    comments = input("Comments (optional): ").strip() or None
    return decision, comments

def get_session_id():
    """
    Connects to the server and retrieves a unique session_id.
    
    Returns:
        str: The session_id issued by the server.
    """

    response = requests.get(f"http://{BASE_URL}/session-id")
    return response.json()["session_id"]

def call_install_embeddings(file_path):
    """
    Calls the /install-embeddings endpoint to upload a new embeddings file.

    Args:
        file_path (str): Path to the embedding file to be uploaded.

    Returns:
        dict: Response message from the server.
    """

    url = f"http://{BASE_URL}/install-embeddings"
    files = {"file": open(file_path, "rb")}
    response = requests.post(url, files=files)
    return response.json()['message']

def call_chat(user_input, session_id=None, auto_approve=False):
    """
    Calls the /chat endpoint to receive a chatbot response based on user input.
    Supports HITL (Human-in-the-Loop) approval flow.

    Args:
        user_input (str): User's input question.
        session_id (str, optional): The unique session ID issued via WebSocket.
        auto_approve (bool): If True, automatically approve all HITL requests.

    Returns:
        str: Chatbot response message.
    """

    headers = {"Content-Type": "application/json"}
    payload = {
        "user_input": user_input,
        "session_id": session_id
    }
    response = requests.post(f"http://{BASE_URL}/chat", headers=headers, data=json.dumps(payload))
    result = response.json()
    
    # HITL processing loop
    while result.get('status') == 'pending_approval':
        tool_name = result.get('tool_name', 'unknown')
        arguments = result.get('arguments', {})
        intermediate = result.get('intermediate_response', '')
        
        # Show previous hitl_decision if exists (chained approvals)
        if result.get('hitl_decision'):
            prev = result['hitl_decision']
            print(f"\n[Previous HITL] {prev['decision'].upper()}: {prev['tool_name']} - {prev.get('comments', 'Unspecified')}")
        
        if intermediate:
            print(f"\n{intermediate}")
        
        print(f"\n{'='*64}")
        print(f"[HITL] Approval Required: {tool_name}")
        print(f"Arguments: {json.dumps(arguments, ensure_ascii=False, indent=2)}")
        print(f"{'='*64}")
        
        decision, comments = _prompt_hitl_decision(auto_approve=auto_approve)
        
        approval_payload = {
            "session_id": session_id,
            "decision": decision,
            "comments": comments
        }
        approval_response = requests.post(
            f"http://{BASE_URL}/approve",
            headers=headers,
            data=json.dumps(approval_payload)
        )
        result = approval_response.json()
        
        # Display hitl_decision from response
        if result.get('hitl_decision'):
            hitl = result['hitl_decision']
            decision_str = hitl['decision'].upper()
            tool = hitl['tool_name']
            cmt = hitl.get('comments', 'Unspecified')
            print(f"\n[HITL Result] {decision_str}: {tool} (Comments: {cmt})")
        
        if result.get('status') == 'rejected':
            return result.get('response', 'Action rejected.')
    
    # Display final hitl_decision if exists
    if result.get('hitl_decision'):
        hitl = result['hitl_decision']
        print(f"\n[HITL Final] {hitl['decision'].upper()}: {hitl['tool_name']} (Comments: {hitl.get('comments', 'Unspecified')})")
    
    return result.get('response', '')

def call_approve(session_id, decision=None, comments=None, approved=None):
    """
    Calls the /approve endpoint to resolve a pending HITL action.

    Args:
        session_id (str): The unique session ID.
        decision (str, optional): One of "approved", "rejected", or "skipped_continue".
                        Short aliases are accepted: "a", "r", "s".
                        Defaults to "approved" when both decision and approved are omitted.
        comments (str): Optional comments (None if empty).
        approved (bool, optional): Backward-compatible flag. True = approved, False = rejected.

    Returns:
        dict: Response from the server including hitl_decision.
    """

    headers = {"Content-Type": "application/json"}
    normalized_decision = _normalize_hitl_decision(decision, approved=approved)
    payload = {
        "session_id": session_id,
        "decision": normalized_decision,
        "comments": comments
    }
    response = requests.post(f"http://{BASE_URL}/approve", headers=headers, data=json.dumps(payload))
    result = response.json()
    
    # Display hitl_decision
    if result.get('hitl_decision'):
        hitl = result['hitl_decision']
        print(f"\n[HITL] {hitl['decision'].upper()}: {hitl['tool_name']} (Comments: {hitl.get('comments', 'Unspecified')})")
    
    return result

def call_set_hitl(auto_approval=False):
    """
    Configures HITL settings on the server.

    Args:
        auto_approval (bool): If True, HITL is disabled (all functions auto-execute).
                             If False, HITL is enabled (critical functions require approval).

    Returns:
        dict: Response message from the server.
    """

    headers = {"Content-Type": "application/json"}
    payload = {"auto_approval": auto_approval}
    response = requests.post(f"http://{BASE_URL}/set-hitl", headers=headers, data=json.dumps(payload))
    return response.json()

def call_get_hitl_status():
    """
    Gets the current HITL configuration from the server.

    Returns:
        dict: Current HITL settings including auto_approval and safe_keywords.
    """

    response = requests.get(f"http://{BASE_URL}/hitl-status")
    return response.json()

def call_chat_stream(user_input, session_id=None, auto_approve=False):
    """
    Connects to the /chat-stream WebSocket endpoint and streams GPT response in real-time.
    Supports HITL (Human-in-the-Loop) approval flow.

    Args:
        user_input (str): User input to send to the chatbot.
        session_id (str, optional): Existing session ID. If None, the server will create one.
        auto_approve (bool): If True, automatically approve all HITL requests.

    Yields:
        str: A partial chunk of the chatbot response.
    """

    try:
        ws = websocket.create_connection(f"ws://{BASE_URL}/chat-stream", timeout=60)

        # Prepare and send the first message (ChatRequest format)
        init_payload = {
            "type": "chat",
            "user_input": user_input,
            "session_id": session_id,
            "user_history_select": ""
        }
        ws.send(json.dumps(init_payload))

        # Stream response from server
        while True:
            try:
                result = ws.recv()
                
                # Check for HITL approval request
                if result.startswith('{"status": "pending_approval"'):
                    hitl_data = json.loads(result)
                    tool_name = hitl_data.get('tool_name', 'unknown')
                    arguments = hitl_data.get('arguments', {})
                    intermediate = hitl_data.get('intermediate_response', '')
                    
                    # Show previous hitl_decision if exists
                    if hitl_data.get('hitl_decision'):
                        prev = hitl_data['hitl_decision']
                        print(f"\n[Previous HITL] {prev['decision'].upper()}: {prev['tool_name']} - {prev.get('comments', 'Unspecified')}")
                    
                    if intermediate:
                        yield f"\n{intermediate}"
                    
                    print()  # newline before HITL prompt
                    print('='*64)
                    print(f"[HITL] Approval Required: {tool_name}")
                    print(f"Arguments: {json.dumps(arguments, ensure_ascii=False, indent=2)}")
                    print('='*64)
                    
                    decision, comments = _prompt_hitl_decision(auto_approve=auto_approve)
                    
                    approval_payload = {
                        "type": "approve",
                        "session_id": session_id,
                        "decision": decision,
                        "comments": comments
                    }
                    ws.send(json.dumps(approval_payload))
                    continue
                
                if result == "__END__":
                    break
                yield result
            except websocket._exceptions.WebSocketTimeoutException:
                break  # No more data from server
            except websocket._exceptions.WebSocketConnectionClosedException:
                break  # Connection closed by server
    except Exception as e:
        yield f"[Error] WebSocket connection failed: {e}"
    finally:
        try:
            ws.close()
        except:
            pass
