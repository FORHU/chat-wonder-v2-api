"""Smoke tests for juris.ph MCP client (mocked HTTP)."""

import json
import unittest
from unittest.mock import MagicMock, patch

from juris_mcp.client import JurisMcpClient, JurisMcpError


class JurisMcpClientTests(unittest.TestCase):
    def test_call_tool_unwraps_json_content(self):
        client = JurisMcpClient(url="https://juris.ph/mcp")
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps({"results": [{"id": "abc", "url": "https://juris.ph/case/abc"}]}),
                    }
                ]
            },
        }
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = payload
        with patch.object(client._session, "post", return_value=mock_resp) as post:
            out = client.call_tool("search_jurisprudence", {"query": "test", "limit": 1})
        self.assertEqual(out["results"][0]["id"], "abc")
        post.assert_called_once()

    def test_call_tool_retries_once_on_transport_error(self):
        import requests

        client = JurisMcpClient(url="https://juris.ph/mcp")
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
            with patch("juris_mcp.client.time.sleep"):
                out = client.call_tool("search_republic_acts", {"query": "safe spaces"})
        self.assertEqual(out["results"], [])
        self.assertEqual(post.call_count, 2)

    def test_rpc_error_raises(self):
        client = JurisMcpClient(url="https://juris.ph/mcp")
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "jsonrpc": "2.0",
            "id": 1,
            "error": {"message": "nope"},
        }
        with patch.object(client._session, "post", return_value=mock_resp):
            with self.assertRaises(JurisMcpError):
                client.call_tool("get_case", {"id": "x"})


if __name__ == "__main__":
    unittest.main()
