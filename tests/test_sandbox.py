"""Tests for server.sandbox.Sandbox — path logic and disabled mode.

These tests run with sandbox.enabled=False to avoid Docker dependency.
"""

import asyncio
import json
import os
from unittest.mock import patch

import pytest

from server.sandbox import (
    Sandbox,
    _host_gid,
    _host_uid,
    _inject_oauth_credentials,
    _skip_special_files,
)


@pytest.fixture
def sandbox(tmp_path):
    """A Sandbox with Docker disabled."""
    tools = tmp_path / "tools"
    tools.mkdir()
    sb = Sandbox(str(tools), enabled=False)
    yield sb
    sb.cleanup()


class TestSandboxPaths:
    def test_workspace_disabled(self, sandbox):
        sandbox.set_benchmark_path("/some/bench")
        # set_benchmark_path calls os.path.abspath, so compare against that
        assert sandbox.workspace == os.path.abspath("/some/bench")

    def test_tools_mount_disabled(self, sandbox):
        assert sandbox.tools_mount == sandbox.tools_dir

    def test_results_dir_is_temp(self, sandbox):
        assert os.path.isdir(sandbox.results_dir)

    def test_result_file_missing(self, sandbox):
        assert sandbox.result_file("nonexistent.json") is None

    def test_result_file_exists(self, sandbox):
        p = os.path.join(sandbox.results_dir, "output.json")
        with open(p, "w") as f:
            f.write("{}")
        assert sandbox.result_file("output.json") == p

    def test_set_dirs(self, sandbox, tmp_path):
        out = tmp_path / "output"
        hacks = tmp_path / "hacks"
        out.mkdir()
        hacks.mkdir()
        sandbox.set_dirs(str(out), str(hacks))
        assert sandbox._output_dir == str(out)
        assert sandbox._jacks_dir == str(hacks)


class TestSandboxCleanup:
    def test_cleanup_removes_results_dir(self, tmp_path):
        tools = tmp_path / "tools"
        tools.mkdir()
        sb = Sandbox(str(tools), enabled=False)
        results_dir = sb.results_dir
        assert os.path.isdir(results_dir)
        sb.cleanup()
        assert not os.path.isdir(results_dir)

    def test_double_cleanup_safe(self, tmp_path):
        tools = tmp_path / "tools"
        tools.mkdir()
        sb = Sandbox(str(tools), enabled=False)
        sb.cleanup()
        sb.cleanup()  # should not raise


class TestHostUidHelpers:
    def test_host_uid_returns_int(self):
        assert isinstance(_host_uid(), int)

    def test_host_gid_returns_int(self):
        assert isinstance(_host_gid(), int)

    def test_host_uid_positive(self):
        assert _host_uid() >= 0

    def test_host_gid_positive(self):
        assert _host_gid() >= 0


class TestNetworkCapArgs:
    def test_ai_container_has_net_caps(self, sandbox):
        # Force enabled so _base_docker_args runs the network branch
        sandbox.enabled = True
        sandbox._claude_dir = str(sandbox.tools_dir)
        args = sandbox._base_docker_args(network=True, ai=False)
        assert "--cap-add" in args
        cap_pairs = [
            args[i + 1] for i, a in enumerate(args) if a == "--cap-add"
        ]
        assert "NET_RAW" in cap_pairs
        assert "NET_BIND_SERVICE" in cap_pairs

    def test_static_container_has_no_network(self, sandbox):
        sandbox.enabled = True
        args = sandbox._base_docker_args(network=False)
        assert "--network" in args
        assert args[args.index("--network") + 1] == "none"
        # No cap-add for NET_RAW on no-network containers
        cap_pairs = [
            args[i + 1] for i, a in enumerate(args) if a == "--cap-add"
        ]
        assert "NET_RAW" not in cap_pairs


class TestLivenessCheck:
    def test_is_alive_returns_false_when_no_container(self, sandbox):
        result = asyncio.get_event_loop().run_until_complete(
            sandbox._is_container_alive()
        )
        assert result is False

    def test_install_extras_noop_when_no_container(self, sandbox):
        # Should return without error when container_id is None
        asyncio.get_event_loop().run_until_complete(
            sandbox.install_extras(["jq"])
        )

    def test_install_extras_noop_when_empty_list(self, tmp_path):
        tools = tmp_path / "tools"
        tools.mkdir()
        sb = Sandbox(str(tools), enabled=False)
        sb._container_id = "fake-id"
        asyncio.get_event_loop().run_until_complete(
            sb.install_extras([])
        )
        sb._container_id = None
        sb.cleanup()


class TestCredentialRefresh:
    def test_refresh_task_none_at_init(self, sandbox):
        assert sandbox._refresh_task is None

    def test_cleanup_cancels_refresh_task(self, tmp_path):
        tools = tmp_path / "tools"
        tools.mkdir()
        sb = Sandbox(str(tools), enabled=False)

        async def _run():
            # Plant a fake long-running task
            sb._refresh_task = asyncio.create_task(asyncio.sleep(9999))
            sb.cleanup()
            assert sb._refresh_task is None

        asyncio.get_event_loop().run_until_complete(_run())

    def test_refresh_loop_exits_when_container_gone(self, tmp_path):
        tools = tmp_path / "tools"
        tools.mkdir()
        sb = Sandbox(str(tools), enabled=False)
        sb._container_id = "alive"
        sb._claude_dir = str(tmp_path)

        cred_calls = []

        def fake_extract():
            cred_calls.append(1)
            return {"claudeAiOauth": {"accessToken": f"tok-{len(cred_calls)}"}}

        async def _run():
            with patch("server.sandbox._extract_claude_credentials", fake_extract):
                # Loop with tiny interval; clear container_id after first tick
                task = asyncio.create_task(
                    sb._refresh_credentials_loop(interval=0.01)
                )
                await asyncio.sleep(0.05)
                sb._container_id = None   # signal loop to stop
                await asyncio.wait_for(task, timeout=1.0)

            # Credentials file should have been written at least once
            dest = tmp_path / ".claude.json"
            assert dest.exists()
            data = json.loads(dest.read_text())
            assert data["claudeAiOauth"]["accessToken"].startswith("tok-")
            assert len(cred_calls) >= 1

        asyncio.get_event_loop().run_until_complete(_run())
        sb.cleanup()

    def test_refresh_loop_updates_credentials_file(self, tmp_path):
        """The refresh must also rewrite the file the container CLI reads."""
        tools = tmp_path / "tools"
        tools.mkdir()
        sb = Sandbox(str(tools), enabled=False)
        sb._container_id = "alive"
        sb._claude_dir = str(tmp_path)

        async def _run():
            with patch("server.sandbox._extract_claude_credentials",
                       lambda: {"claudeAiOauth": {"accessToken": "tok-fresh"}}):
                task = asyncio.create_task(
                    sb._refresh_credentials_loop(interval=0.01)
                )
                await asyncio.sleep(0.05)
                sb._container_id = None
                await asyncio.wait_for(task, timeout=1.0)

            creds = tmp_path / ".claude" / ".credentials.json"
            assert creds.exists()
            data = json.loads(creds.read_text())
            assert data["claudeAiOauth"]["accessToken"] == "tok-fresh"

        # asyncio.run rather than get_event_loop().run_until_complete(): the
        # latter is what the older tests here use and it fails once another
        # module in the suite has closed the loop.
        asyncio.run(_run())
        sb.cleanup()


class TestSkipSpecialFiles:
    """copytree must not choke on the sockets a live home directory holds."""

    def test_skips_socket(self, tmp_path, monkeypatch):
        import socket
        # AF_UNIX paths are capped near 104 bytes, below pytest's tmp_path
        # length, so bind relative from inside the directory.
        monkeypatch.chdir(tmp_path)
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            sock.bind("ipc.sock")
            (tmp_path / "config.json").write_text("{}")
            skipped = _skip_special_files(str(tmp_path),
                                          ["ipc.sock", "config.json"])
        finally:
            sock.close()
        assert skipped == {"ipc.sock"}

    def test_skips_fifo(self, tmp_path):
        os.mkfifo(str(tmp_path / "pipe"))
        (tmp_path / "keep.txt").write_text("x")
        assert _skip_special_files(str(tmp_path), ["pipe", "keep.txt"]) == {"pipe"}

    def test_keeps_regular_files_and_dirs(self, tmp_path):
        (tmp_path / "a.json").write_text("{}")
        (tmp_path / "sub").mkdir()
        assert _skip_special_files(str(tmp_path), ["a.json", "sub"]) == set()

    def test_unreadable_entry_is_skipped(self, tmp_path):
        assert _skip_special_files(str(tmp_path), ["does-not-exist"]) == {
            "does-not-exist"}


class TestInjectOAuthCredentials:
    def test_writes_both_locations(self, tmp_path):
        creds = {"claudeAiOauth": {"accessToken": "tok-1"}}
        _inject_oauth_credentials(str(tmp_path), creds)

        config = json.loads((tmp_path / ".claude.json").read_text())
        assert config["claudeAiOauth"]["accessToken"] == "tok-1"

        cred_file = tmp_path / ".claude" / ".credentials.json"
        assert json.loads(cred_file.read_text()) == creds

    def test_credentials_file_is_owner_only(self, tmp_path):
        _inject_oauth_credentials(str(tmp_path), {"claudeAiOauth": {}})
        mode = (tmp_path / ".claude" / ".credentials.json").stat().st_mode
        assert mode & 0o077 == 0

    def test_preserves_unrelated_config_keys(self, tmp_path):
        (tmp_path / ".claude.json").write_text(json.dumps({"theme": "dark"}))
        _inject_oauth_credentials(str(tmp_path),
                                  {"claudeAiOauth": {"accessToken": "tok-2"}})
        config = json.loads((tmp_path / ".claude.json").read_text())
        assert config["theme"] == "dark"
        assert config["claudeAiOauth"]["accessToken"] == "tok-2"

    def test_corrupt_config_is_replaced_not_fatal(self, tmp_path):
        (tmp_path / ".claude.json").write_text("not json{{")
        _inject_oauth_credentials(str(tmp_path), {"claudeAiOauth": {"a": 1}})
        assert json.loads((tmp_path / ".claude.json").read_text())["claudeAiOauth"]
