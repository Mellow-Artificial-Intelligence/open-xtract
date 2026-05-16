"""Extract structured meeting notes from an audio recording using openextract."""

import sys
from datetime import date

from pydantic import BaseModel

from openextract import extract


class ActionItem(BaseModel):
    description: str
    owner: str
    due: date | None = None


class Meeting(BaseModel):
    attendees: list[str]
    date: date
    summary: str
    decisions: list[str]
    action_items: list[ActionItem]


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: uv run python examples/meeting_notes.py <audio-file>")
        sys.exit(1)

    input_file = sys.argv[1]

    meeting = extract(
        schema=Meeting,
        model="openai:gpt-5",
        input_file=input_file,
        instructions=(
            "Listen to the meeting audio and produce structured notes. Include "
            "every attendee, the meeting date, a concise summary, the decisions "
            "that were made, and every action item with its owner and due date "
            "when mentioned."
        ),
    )

    print(meeting.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
