"""Extract structured meeting notes from an audio recording."""

import sys
from datetime import date

from pydantic import BaseModel

from examples._shared import require_input, xai_model
from openextract import extract


class ActionItem(BaseModel):
    description: str
    owner: str
    due: date | None = None


class Meeting(BaseModel):
    attendees: list[str]
    meeting_date: date | None = None
    summary: str
    decisions: list[str]
    action_items: list[ActionItem]


def main() -> None:
    input_file = require_input(
        sys.argv,
        "Usage: uv run python examples/audio/meeting_notes.py <audio-file>",
    )

    meeting = extract(
        schema=Meeting,
        model=xai_model(),
        input_file=input_file,
        instructions=(
            "Listen to the meeting audio and produce structured notes. Include "
            "every attendee, the meeting date when mentioned, a concise summary, "
            "decisions made, and action items with owners and due dates."
        ),
    )

    print(meeting.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
