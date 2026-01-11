"""Demo script showing durable extraction with Temporal."""

from pydantic import BaseModel

from open_xtract import extract, stop_temporal


class Article(BaseModel):
    """Extracted article information."""

    title: str
    author: str
    summary: str
    key_points: list[str]


def main():
    # Example: Extract information from a Wikipedia article
    url = "https://cdn.openai.com/business-guides-and-resources/a-practical-guide-to-building-agents.pdf"

    print("Starting durable extraction...")
    print(f"URL: {url}")
    print()

    result = extract(
        schema=Article,
        model="openai:gpt-5.2",
        url=url,
        instructions="Extract the article title, author, a brief summary, and 3-5 key points.",
        durable=True,
        temporal_ui=True,
    )

    print("Extraction complete!")
    print()
    print(f"Title: {result.title}")
    print(f"Author: {result.author}")
    print(f"Summary: {result.summary}")
    print()
    print("Key Points:")
    for i, point in enumerate(result.key_points, 1):
        print(f"  {i}. {point}")


if __name__ == "__main__":
    main()

    # Optional: uncomment to stop Temporal services when done
    stop_temporal()
