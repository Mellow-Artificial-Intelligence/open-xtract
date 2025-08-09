"""Optional demo script for StructuredOutputGenerator.

This file avoids hard dependencies on model provider packages and external API
calls. If `langchain_openai` is available and an API key is configured, it will
run a small demo. Otherwise, it prints a short message with setup steps.
"""

import os


def main() -> None:
    try:
        from langchain_openai import ChatOpenAI  # type: ignore
        from pydantic import BaseModel
        from open_xtract import StructuredOutputGenerator
        from langchain_core.prompts import ChatPromptTemplate
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

    class Citation(BaseModel):
        source_id: int
        quote: str

    class AnnotatedAnswer(BaseModel):
        citations: list[Citation]

    llm = ChatOpenAI(model="gpt-4o-mini")

    generator = StructuredOutputGenerator(
        llm=llm,
        schema=AnnotatedAnswer,
        method="auto",
        strict=True,
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Answer the question using the provided context."),
            ("human", "Question: {question}\nContext:\n{context}"),
        ]
    )

    result = generator.invoke(
        prompt=prompt,
        variables={"question": "How fast are cheetahs?", "context": "..."},
    )
    print(result)


if __name__ == "__main__":
    main()