from .structured_output import StructuredOutputGenerator


def main() -> None:  # Console entrypoint defined in pyproject.toml
    try:
        from .main import main as _main
    except Exception:
        # Fallback no-op if a richer CLI isn't present
        def _main() -> None:
            print("open-xtract: CLI not configured.")

    _main()


__all__ = ["StructuredOutputGenerator", "main"]


