"""Optional demo script using OpenXtract.

This file avoids hard dependencies on external API calls if no API key is set.
If `langchain_openai` is available and an API key is configured, it will run a
small demo; otherwise, it prints setup steps.
"""

import os


def main() -> None:
    try:
        from pydantic import BaseModel
        from open_xtract import OpenXtract
    except Exception:
        print(
            "Demo prerequisites missing. Install optional deps and try again:\n"
            "  pip install langchain-openai\n"
            "Or use the package entrypoint: `open-xtract`\n"
        )
        return

    if not (os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY_PATH")):
        print("Set OPENAI_API_KEY to run the demo. Skipping network call.")
        return

    class Answer(BaseModel):
        text: str

    ox = OpenXtract(model="gpt-5-mini")
    result = ox.extract_text("Summarize cheetah top speed in one sentence.", Answer)
    print(result)


if __name__ == "__main__":
    main()