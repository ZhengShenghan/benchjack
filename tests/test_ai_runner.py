"""Tests for server.ai_runner — stream-json parser and tool summariser."""

import json

import pytest

from server.ai_runner import AIRunner, RateLimitError, _summarise_tool_input


# ── _parse_stream_json_line ─────────────────────────────────────────

class TestParseStreamJsonLine:
    def _parse(self, line: str) -> list[dict]:
        return list(AIRunner._parse_stream_json_line(line))

    def test_assistant_text(self):
        evt = {
            "type": "assistant",
            "message": {"content": [{"type": "text", "text": "Hello world"}]},
        }
        results = self._parse(json.dumps(evt))
        assert len(results) == 1
        assert results[0] == {"msg_type": "text", "text": "Hello world"}

    def test_assistant_tool_use(self):
        evt = {
            "type": "assistant",
            "message": {
                "content": [
                    {
                        "type": "tool_use",
                        "name": "Read",
                        "input": {"file_path": "/foo/bar.py"},
                    }
                ]
            },
        }
        results = self._parse(json.dumps(evt))
        assert len(results) == 1
        assert results[0]["msg_type"] == "tool_call"
        assert results[0]["name"] == "Read"
        assert results[0]["summary"] == "/foo/bar.py"

    def test_user_tool_result(self):
        evt = {
            "type": "user",
            "message": {
                "content": [
                    {"type": "tool_result", "content": "x" * 100},
                ]
            },
        }
        results = self._parse(json.dumps(evt))
        assert len(results) == 1
        assert results[0]["msg_type"] == "tool_result"
        assert results[0]["chars"] == 100

    def test_user_tool_result_list_content(self):
        evt = {
            "type": "user",
            "message": {
                "content": [
                    {"type": "tool_result", "content": ["abc", "de"]},
                ]
            },
        }
        results = self._parse(json.dumps(evt))
        # sum(len(str(c)) for c in ["abc", "de"]) = 3 + 2 = 5
        assert results[0]["chars"] == 5

    def test_system_event_ignored(self):
        evt = {"type": "system", "message": "init"}
        assert self._parse(json.dumps(evt)) == []

    def test_result_event_yields_usage(self):
        # A result event with no usage block still yields a usage event
        # (ai_calls counting relies on one usage event per result), with
        # zeroed tokens and null cost.
        evt = {"type": "result", "result": "final text"}
        results = self._parse(json.dumps(evt))
        assert len(results) == 1
        assert results[0]["msg_type"] == "usage"
        assert results[0]["input_tokens"] == 0
        assert results[0]["output_tokens"] == 0
        assert results[0]["cost_usd"] is None

    def test_result_event_extracts_usage_and_cost(self):
        evt = {
            "type": "result",
            "subtype": "success",
            "duration_ms": 4200,
            "duration_api_ms": 3900,
            "num_turns": 7,
            "total_cost_usd": 0.1234,
            "usage": {
                "input_tokens": 100,
                "output_tokens": 50,
                "cache_creation_input_tokens": 200,
                "cache_read_input_tokens": 300,
            },
            "modelUsage": {"claude-opus-4-8": {"inputTokens": 100}},
            "result": "final text",
            "session_id": "abc123",
        }
        results = self._parse(json.dumps(evt))
        assert len(results) == 1
        u = results[0]
        assert u["msg_type"] == "usage"
        # input_tokens folds cache_read + cache_creation: 100 + 300 + 200 = 600
        assert u["input_tokens"] == 600
        assert u["output_tokens"] == 50
        assert u["cost_usd"] == pytest.approx(0.1234)
        assert u["num_turns"] == 7
        assert u["model"] == "claude-opus-4-8"

    def test_invalid_json_yields_text(self):
        results = self._parse("not json at all")
        assert len(results) == 1
        assert results[0]["msg_type"] == "text"
        assert results[0]["text"] == "not json at all"

    def test_rate_limit_in_text_raises(self):
        evt = {
            "type": "assistant",
            "message": {"content": [{"type": "text", "text": "You've hit your limit"}]},
        }
        with pytest.raises(RateLimitError):
            self._parse(json.dumps(evt))

    def test_rate_limit_in_raw_line_raises(self):
        with pytest.raises(RateLimitError):
            self._parse("You've hit your limit on free messages")

    def test_rate_limit_in_result_raises(self):
        evt = {"type": "result", "result": "You've hit your limit"}
        with pytest.raises(RateLimitError):
            self._parse(json.dumps(evt))

    def test_multiple_content_blocks(self):
        evt = {
            "type": "assistant",
            "message": {
                "content": [
                    {"type": "text", "text": "Analyzing..."},
                    {"type": "tool_use", "name": "Bash", "input": {"command": "ls"}},
                    {"type": "text", "text": "Done."},
                ]
            },
        }
        results = self._parse(json.dumps(evt))
        assert len(results) == 3
        assert results[0]["msg_type"] == "text"
        assert results[1]["msg_type"] == "tool_call"
        assert results[2]["msg_type"] == "text"


# ── _summarise_tool_input ───────────────────────────────────────────

class TestSummariseToolInput:
    def test_read(self):
        assert _summarise_tool_input("Read", {"file_path": "/a/b.py"}) == "/a/b.py"

    def test_grep(self):
        result = _summarise_tool_input("Grep", {"pattern": "TODO", "path": "src/"})
        assert "TODO" in result
        assert "src/" in result

    def test_glob(self):
        assert _summarise_tool_input("Glob", {"pattern": "**/*.py"}) == "**/*.py"

    def test_bash(self):
        assert _summarise_tool_input("Bash", {"command": "ls -la"}) == "ls -la"

    def test_bash_long_truncated(self):
        cmd = "x" * 200
        result = _summarise_tool_input("Bash", {"command": cmd})
        assert len(result) == 120
        assert result.endswith("...")

    def test_edit(self):
        assert _summarise_tool_input("Edit", {"file_path": "/foo.py"}) == "/foo.py"

    def test_write(self):
        assert _summarise_tool_input("Write", {"file_path": "/bar.py"}) == "/bar.py"

    def test_web_fetch(self):
        assert _summarise_tool_input("WebFetch", {"url": "https://example.com"}) == "https://example.com"

    def test_web_search(self):
        assert _summarise_tool_input("WebSearch", {"query": "python async"}) == "python async"

    def test_unknown_tool(self):
        result = _summarise_tool_input("CustomTool", {"a": 1})
        assert '"a"' in result

    def test_unknown_tool_long_truncated(self):
        inp = {"data": "x" * 200}
        result = _summarise_tool_input("CustomTool", inp)
        assert len(result) == 120
        assert result.endswith("...")
