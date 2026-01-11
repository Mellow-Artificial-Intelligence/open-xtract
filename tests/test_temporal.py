"""Tests for Temporal durable execution."""

from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

from open_xtract import extract


class TestDurableExtraction:
    def test_durable_requires_temporal_dependency(self, mocker):
        """Test that durable=True raises ImportError when temporal not installed."""
        import sys

        class TestSchema(BaseModel):
            data: str

        original_modules = sys.modules.copy()
        sys.modules["open_xtract._temporal"] = None

        try:
            with pytest.raises(ImportError) as exc_info:
                extract(
                    schema=TestSchema,
                    model="test-model",
                    url="https://example.com/doc.pdf",
                    instructions="test",
                    durable=True,
                )

            assert "Temporal dependencies not installed" in str(exc_info.value)
        finally:
            if "open_xtract._temporal" in original_modules:
                sys.modules["open_xtract._temporal"] = original_modules["open_xtract._temporal"]
            else:
                sys.modules.pop("open_xtract._temporal", None)

    def test_durable_false_uses_sync_extraction(self, mocker):
        """Test that durable=False uses the normal sync extraction path."""

        class TestSchema(BaseModel):
            title: str

        mock_output = TestSchema(title="Test")
        mock_result = MagicMock()
        mock_result.output = mock_output

        mock_agent_instance = MagicMock()
        mock_agent_instance.run_sync.return_value = mock_result

        mocker.patch("open_xtract._extract.Agent", return_value=mock_agent_instance)

        result = extract(
            schema=TestSchema,
            model="test-model",
            url="https://example.com/doc.pdf",
            instructions="test",
            durable=False,
        )

        assert result == mock_output
        mock_agent_instance.run_sync.assert_called_once()


class TestDockerModule:
    def test_is_temporal_running_returns_false_on_connection_error(self, mocker):
        """Test that is_temporal_running returns False when connection fails."""
        from open_xtract._docker import is_temporal_running

        mocker.patch("socket.create_connection", side_effect=ConnectionRefusedError())

        assert is_temporal_running() is False

    def test_is_temporal_running_returns_true_on_success(self, mocker):
        """Test that is_temporal_running returns True when connection succeeds."""
        from open_xtract._docker import is_temporal_running

        mock_socket = MagicMock()
        mock_socket.__enter__ = MagicMock(return_value=mock_socket)
        mock_socket.__exit__ = MagicMock(return_value=False)
        mocker.patch("socket.create_connection", return_value=mock_socket)

        assert is_temporal_running() is True

    def test_is_temporal_ready_returns_true_when_healthy(self, mocker):
        """Test that is_temporal_ready returns True when container is healthy."""
        from open_xtract._docker import is_temporal_ready

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "healthy\n"
        mocker.patch("subprocess.run", return_value=mock_result)

        assert is_temporal_ready() is True

    def test_is_temporal_ready_returns_false_when_not_healthy(self, mocker):
        """Test that is_temporal_ready returns False when container is not healthy."""
        from open_xtract._docker import is_temporal_ready

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "starting\n"
        mocker.patch("subprocess.run", return_value=mock_result)

        assert is_temporal_ready() is False

    def test_is_docker_available_returns_true_when_docker_running(self, mocker):
        """Test that is_docker_available returns True when Docker is running."""
        from open_xtract._docker import is_docker_available

        mocker.patch("subprocess.run", return_value=MagicMock(returncode=0))

        assert is_docker_available() is True

    def test_is_docker_available_returns_false_when_docker_not_running(self, mocker):
        """Test that is_docker_available returns False when Docker is not running."""
        import subprocess

        from open_xtract._docker import is_docker_available

        mocker.patch("subprocess.run", side_effect=subprocess.CalledProcessError(1, "docker info"))

        assert is_docker_available() is False

    def test_is_docker_available_returns_false_when_docker_not_installed(self, mocker):
        """Test that is_docker_available returns False when Docker is not installed."""
        from open_xtract._docker import is_docker_available

        mocker.patch("subprocess.run", side_effect=FileNotFoundError())

        assert is_docker_available() is False

    def test_start_temporal_server_skips_if_already_running(self, mocker):
        """Test that start_temporal_server does nothing if Temporal is already running."""
        from open_xtract._docker import start_temporal_server

        mocker.patch("open_xtract._docker.is_temporal_ready", return_value=True)
        mock_docker_check = mocker.patch("open_xtract._docker.is_docker_available")

        start_temporal_server()

        mock_docker_check.assert_not_called()

    def test_start_temporal_server_raises_if_docker_unavailable(self, mocker):
        """Test that start_temporal_server raises RuntimeError if Docker unavailable."""
        from open_xtract._docker import start_temporal_server

        mocker.patch("open_xtract._docker.is_temporal_ready", return_value=False)
        mocker.patch("open_xtract._docker.is_docker_available", return_value=False)

        with pytest.raises(RuntimeError) as exc_info:
            start_temporal_server()

        assert "Docker is required" in str(exc_info.value)

    def test_start_temporal_server_starts_new_container(self, mocker):
        """Test that start_temporal_server creates new container if none exists."""
        from open_xtract._docker import start_temporal_server

        mocker.patch("open_xtract._docker.is_temporal_ready", side_effect=[False, False, True])
        mocker.patch("open_xtract._docker.is_docker_available", return_value=True)
        mocker.patch("open_xtract._docker._is_compose_running", return_value=False)

        mock_run = mocker.patch("subprocess.run")
        mock_run.return_value = MagicMock(stdout="", returncode=0)

        mocker.patch("time.sleep")

        start_temporal_server()

        assert mock_run.call_count >= 1

    def test_start_temporal_server_starts_existing_container(self, mocker):
        """Test that start_temporal_server starts existing stopped container."""
        from open_xtract._docker import start_temporal_server

        mocker.patch("open_xtract._docker.is_temporal_ready", side_effect=[False, False, True])
        mocker.patch("open_xtract._docker.is_docker_available", return_value=True)
        mocker.patch("open_xtract._docker._is_compose_running", return_value=True)

        mock_run = mocker.patch("subprocess.run")
        mock_run.return_value = MagicMock(stdout="", returncode=0)

        mocker.patch("time.sleep")

        start_temporal_server()

        calls = mock_run.call_args_list
        start_call = [c for c in calls if "start" in str(c)]
        assert len(start_call) > 0

    def test_start_temporal_server_raises_on_timeout(self, mocker):
        """Test that start_temporal_server raises RuntimeError on timeout."""
        from open_xtract._docker import start_temporal_server

        mocker.patch("open_xtract._docker.is_temporal_ready", return_value=False)
        mocker.patch("open_xtract._docker.is_docker_available", return_value=True)
        mocker.patch("open_xtract._docker._is_compose_running", return_value=False)

        mock_run = mocker.patch("subprocess.run")
        mock_run.return_value = MagicMock(stdout="", returncode=0)

        mocker.patch("time.sleep")

        with pytest.raises(RuntimeError) as exc_info:
            start_temporal_server(timeout_seconds=2)

        assert "failed to start" in str(exc_info.value)
