"""Smoke tests for the Korean Law MCP client (mocked HTTP)."""

import json
import unittest
from unittest.mock import MagicMock, patch

from korean_law_mcp.client import KoreanLawMcpClient, KoreanLawMcpError


class KoreanLawMcpClientTests(unittest.TestCase):
    def test_call_tool_unwraps_json_content(self):
        client = KoreanLawMcpClient(url="https://mcp.gomdori.app/law")
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps({"results": [{"lawId": "001234", "lawName": "민법"}]}),
                    }
                ]
            },
        }
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = payload
        with patch.object(client._session, "post", return_value=mock_resp) as post:
            out = client.call_tool("search_law", {"query": "민법"})
        self.assertEqual(out["results"][0]["lawId"], "001234")
        post.assert_called_once()

    def test_call_tool_retries_once_on_transport_error(self):
        import requests

        client = KoreanLawMcpClient(url="https://mcp.gomdori.app/law")
        ok_payload = {
            "jsonrpc": "2.0",
            "id": 2,
            "result": {"content": [{"type": "text", "text": '{"results":[]}'}]},
        }
        ok_resp = MagicMock()
        ok_resp.raise_for_status = MagicMock()
        ok_resp.json.return_value = ok_payload

        with patch.object(
            client._session,
            "post",
            side_effect=[requests.ConnectionError("boom"), ok_resp],
        ) as post:
            with patch("mcp_client.time.sleep"):
                out = client.call_tool("search_decisions", {"domain": "precedent", "query": "negligence"})
        self.assertEqual(out["results"], [])
        self.assertEqual(post.call_count, 2)

    def test_rpc_error_raises(self):
        client = KoreanLawMcpClient(url="https://mcp.gomdori.app/law")
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "jsonrpc": "2.0",
            "id": 1,
            "error": {"message": "nope"},
        }
        with patch.object(client._session, "post", return_value=mock_resp):
            with self.assertRaises(KoreanLawMcpError):
                client.call_tool("get_law_text", {"mst": "x"})

    def test_api_key_sent_as_header_when_set(self):
        client = KoreanLawMcpClient(url="https://mcp.gomdori.app/law", api_key="test-key")
        self.assertEqual(client._session.headers.get("apikey"), "test-key")

    def test_no_api_key_header_by_default(self):
        client = KoreanLawMcpClient(url="https://mcp.gomdori.app/law")
        self.assertNotIn("apikey", client._session.headers)


if __name__ == "__main__":
    unittest.main()
