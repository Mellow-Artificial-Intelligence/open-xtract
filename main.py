"""Optional demo script using OpenXtract.

This file avoids hard dependencies on external API calls if no API key is set.
If `langchain_openai` is available and an API key is configured, it will run a
small demo; otherwise, it prints setup steps.
"""

import os


def main() -> None:
    try:
        from open_xtract import OpenXtract
        from pydantic import BaseModel
    except Exception:
        print(
            "Demo prerequisites missing. Install optional deps and try again:\n"
            "  pip install langchain-openai\n"
            "Or use the package entrypoint: `open-xtract`\n"
        )
        return

    api_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Set ANTHROPIC_API_KEY or OPENAI_API_KEY to run the demo. Skipping network call.")
        return

    class Answer(BaseModel):
        text: str

    # Use Claude if Anthropic key is available, otherwise OpenAI
    model = "claude-opus-4-1-20250805" if os.getenv("ANTHROPIC_API_KEY") else "gpt-5-mini"
    ox = OpenXtract(model=model)
    result = ox.extract("Summarize cheetah top speed in one sentence.", Answer)
    print(result)


if __name__ == "__main__":
    main()
