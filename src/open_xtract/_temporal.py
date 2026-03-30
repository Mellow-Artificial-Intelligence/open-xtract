"""Temporal durable execution support for extraction."""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from uuid import uuid4

from dotenv import load_dotenv
from pydantic import BaseModel
from pydantic_ai import Agent, AudioUrl, DocumentUrl, ImageUrl, VideoUrl
from temporalio import activity, workflow
from temporalio.client import Client
from temporalio.worker import Worker

from ._docker import start_temporal_server
from ._extract import AUDIO_EXTENSIONS, IMAGE_EXTENSIONS, VIDEO_EXTENSIONS, _validate_url

# Load environment variables from .env file
load_dotenv(Path.cwd() / ".env")

TASK_QUEUE = "open-xtract"


@dataclass
class ExtractionParams:
    """Parameters for extraction workflow."""

    schema_json: str
    schema_name: str
    model: str
    url: str
    instructions: str


def _get_media_url(url: str):
    """Determine the appropriate media URL type based on file extension."""
    parsed = urlparse(url)
    ext = os.path.splitext(parsed.path)[1].lower()

    if ext in IMAGE_EXTENSIONS:
        return ImageUrl(url=url)
    elif ext in AUDIO_EXTENSIONS:
        return AudioUrl(url=url)
    elif ext in VIDEO_EXTENSIONS:
        return VideoUrl(url=url)
    else:
        return DocumentUrl(url=url)


@activity.defn
async def extract_activity(params: ExtractionParams) -> dict[str, Any]:
    """Activity that performs the actual extraction."""
    # Include schema in instructions so the LLM knows exact field names
    full_instructions = f"""{params.instructions}

You must return a JSON object matching this exact schema:
{params.schema_json}"""

    agent = Agent(params.model, instructions=full_instructions, output_type=dict)
    media_url = _get_media_url(params.url)

    result = await agent.run(
        [
            "Extract the requested information from this document.",
            media_url,
        ]
    )
    return result.output


@workflow.defn
class ExtractionWorkflow:
    """Workflow for durable extraction."""

    @workflow.run
    async def run(self, params: ExtractionParams) -> dict[str, Any]:
        """Execute extraction with durable guarantees."""
        return await workflow.execute_activity(
            extract_activity,
            params,
            start_to_close_timeout=workflow.timedelta(minutes=5),
        )


async def run_durable_extraction(
    schema: type[BaseModel],
    model: str,
    url: str,
    instructions: str,
    *,
    temporal_ui: bool = True,
) -> BaseModel:
    """
    Run extraction with durable execution via Temporal.

    Automatically starts Temporal server via Docker if not running.

    Args:
        schema: A Pydantic model class defining the expected output structure.
        model: The model identifier (e.g., 'google-gla:gemini-3-flash-preview').
        url: The URL of the document, image, audio, or video to extract from.
        instructions: Instructions for the LLM on what to extract.
        temporal_ui: If True, start the Temporal UI alongside the server.

    Returns:
        An instance of the schema populated with extracted data.

    Raises:
        RuntimeError: If Docker or Temporal is not available.
    """
    _validate_url(url)

    start_temporal_server(with_ui=temporal_ui)

    client = await Client.connect("localhost:7233")

    params = ExtractionParams(
        schema_json=json.dumps(schema.model_json_schema()),
        schema_name=schema.__name__,
        model=model,
        url=url,
        instructions=instructions,
    )

    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[ExtractionWorkflow],
        activities=[extract_activity],
    ):
        result = await client.execute_workflow(
            ExtractionWorkflow.run,
            params,
            id=f"extract-{uuid4()}",
            task_queue=TASK_QUEUE,
        )

    return schema.model_validate(result)
