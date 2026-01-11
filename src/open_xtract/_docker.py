"""Docker management for Temporal server with PostgreSQL and UI."""

import socket
import subprocess
import time
from pathlib import Path

TEMPORAL_PORT = 7233
TEMPORAL_UI_PORT = 8080
COMPOSE_PROJECT = "open-xtract-temporal"
COMPOSE_FILE = Path(__file__).parent / "docker-compose.temporal.yml"


def is_temporal_running() -> bool:
    """Check if Temporal is accessible on localhost:7233."""
    try:
        with socket.create_connection(("localhost", TEMPORAL_PORT), timeout=1):
            return True
    except (TimeoutError, ConnectionRefusedError, OSError):
        return False


def is_temporal_ready() -> bool:
    """Check if Temporal container is healthy and ready."""
    try:
        result = subprocess.run(
            [
                "docker",
                "inspect",
                "--format",
                "{{.State.Health.Status}}",
                "open-xtract-temporal",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.returncode == 0 and result.stdout.strip() == "healthy"
    except (subprocess.TimeoutExpired, subprocess.SubprocessError):
        return False


def is_docker_available() -> bool:
    """Check if Docker daemon is running."""
    try:
        subprocess.run(["docker", "info"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def _is_compose_running() -> bool:
    """Check if the docker-compose stack is running."""
    result = subprocess.run(
        ["docker", "compose", "-f", str(COMPOSE_FILE), "-p", COMPOSE_PROJECT, "ps", "-q"],
        capture_output=True,
        text=True,
    )
    return bool(result.stdout.strip())


def start_temporal_server(timeout_seconds: int = 60, *, with_ui: bool = True) -> None:
    """
    Start Temporal server with PostgreSQL via Docker Compose.

    Args:
        timeout_seconds: Maximum time to wait for Temporal to start.
        with_ui: If True (default), also start the Temporal UI on port 8080.

    Raises:
        RuntimeError: If Docker is not available or Temporal fails to start.
    """
    if is_temporal_ready():
        if with_ui:
            print(f"\n  Temporal UI: http://localhost:{TEMPORAL_UI_PORT}\n")
        return

    if not is_docker_available():
        raise RuntimeError(
            "Docker is required for durable execution. "
            "Please install and start Docker, then try again."
        )

    if with_ui:
        print("Starting Temporal server with PostgreSQL and UI...")
    else:
        print("Starting Temporal server with PostgreSQL...")

    # Build the list of services to start
    services = ["postgresql", "temporal"]
    if with_ui:
        services.append("temporal-ui")

    if _is_compose_running():
        subprocess.run(
            ["docker", "compose", "-f", str(COMPOSE_FILE), "-p", COMPOSE_PROJECT, "start"]
            + services,
            check=True,
            capture_output=True,
        )
    else:
        subprocess.run(
            [
                "docker",
                "compose",
                "-f",
                str(COMPOSE_FILE),
                "-p",
                COMPOSE_PROJECT,
                "up",
                "-d",
            ]
            + services,
            check=True,
            capture_output=True,
        )

    print("Waiting for Temporal to be ready...")
    for i in range(timeout_seconds):
        if is_temporal_ready():
            if with_ui:
                print(f"\n  Temporal UI: http://localhost:{TEMPORAL_UI_PORT}\n")
            else:
                print("\n  Temporal server ready.\n")
            return
        if i % 10 == 0 and i > 0:
            print(f"  Still waiting... ({i}s)")
        time.sleep(1)

    raise RuntimeError(f"Temporal server failed to start within {timeout_seconds} seconds")


def stop_temporal_server() -> None:
    """Stop the Temporal server stack."""
    if not is_docker_available():
        return

    subprocess.run(
        ["docker", "compose", "-f", str(COMPOSE_FILE), "-p", COMPOSE_PROJECT, "stop"],
        capture_output=True,
    )
    print("Temporal server stopped.")
