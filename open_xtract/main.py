"""OpenXtract - Extract structured data from documents using Claude Code."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, Literal, TypeVar

from claude_agent_sdk import ClaudeAgentOptions, ResultMessage, query
from pydantic import BaseModel

from .exceptions import ExtractionError, InputError

T = TypeVar("T", bound=BaseModel)

ModelType = Literal["claude-sonnet-4-5", "claude-opus-4-5", "claude-haiku-4-5"]
PermissionModeType = Literal["default", "acceptEdits", "plan", "bypassPermissions"]


@dataclass
class UsageStats:
    """Token usage statistics."""

    input_tokens: int
    output_tokens: int
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        """Total tokens used."""
        return self.input_tokens + self.output_tokens


@dataclass
class ExtractionResult(Generic[T]):
    """Result of an extraction operation with metadata."""

    data: T
    """The extracted data matching the schema."""

    model: str | None
    """Model used for extraction."""

    duration_ms: int
    """Total duration in milliseconds."""

    duration_api_ms: int
    """API call duration in milliseconds."""

    num_turns: int
    """Number of conversation turns."""

    session_id: str
    """Session identifier."""

    cost_usd: float | None
    """Total cost in USD."""

    usage: UsageStats | None
    """Token usage statistics."""


class OpenXtract:
    """Extract structured data from documents using Claude Code.

    Supports text files, images (PNG, JPG, etc.), and PDFs.
    """

    def __init__(
        self,
        model: ModelType | None = None,
        permission_mode: PermissionModeType = "acceptEdits",
        system_prompt: str | None = None,
    ) -> None:
        """Initialize the extractor.

        Args:
            model: Optional model to use. Valid values:
                   "claude-sonnet-4-5", "claude-opus-4-5", "claude-haiku-4-5".
            permission_mode: Permission mode for Claude Code. Defaults to "acceptEdits".
            system_prompt: Optional system prompt to guide extraction behavior.
        """
        self._model = model
        self._permission_mode = permission_mode
        self._system_prompt = system_prompt

    async def extract(self, file_path: str | Path, schema: type[T]) -> ExtractionResult[T]:
        """Extract structured data from a file.

        Args:
            file_path: Path to the file (text, image, or PDF).
            schema: Pydantic model class defining the expected output structure.

        Returns:
            ExtractionResult containing the extracted data and metadata.

        Raises:
            InputError: If the file doesn't exist.
            ExtractionError: If extraction fails or output doesn't match schema.
        """
        path = Path(file_path)
        if not path.exists():
            raise InputError(
                f"File not found: {file_path}",
                suggestions=[
                    "Check that the file path is correct",
                    "Ensure the file exists before calling extract()",
                ],
            )

        # Simple prompt - let Claude Code handle the rest
        prompt = f"Read '{path.absolute()}' and extract data matching the output schema."

        # Convert Pydantic schema to JSON schema
        json_schema = schema.model_json_schema()

        # Configure options
        options = ClaudeAgentOptions(
            allowed_tools=["Read"],
            permission_mode=self._permission_mode,
            output_format={"type": "json_schema", "schema": json_schema},
        )

        if self._model:
            options.model = self._model

        if self._system_prompt:
            options.system_prompt = self._system_prompt

        # Run extraction
        result_data: dict | None = None
        result_message: ResultMessage | None = None
        model_used: str | None = None

        async for message in query(prompt=prompt, options=options):
            if isinstance(message, ResultMessage):
                result_message = message
                if message.is_error:
                    raise ExtractionError(
                        f"Extraction failed: {message.result or 'Unknown error'}",
                        suggestions=[
                            "Check that the file is readable",
                            "Ensure the schema matches the document structure",
                        ],
                    )
                # Structured output is in the structured_output attribute
                if hasattr(message, "structured_output") and message.structured_output:
                    result_data = message.structured_output
            else:
                # Capture model from AssistantMessage
                if hasattr(message, "model") and message.model:
                    model_used = message.model
                # Also capture from ToolUseBlock with StructuredOutput name
                if hasattr(message, "content"):
                    for block in message.content:
                        if hasattr(block, "name") and block.name == "StructuredOutput":
                            result_data = block.input

        if not result_data:
            raise ExtractionError(
                "No result returned from extraction",
                suggestions=["Check that the file contains extractable content"],
            )

        # Validate against schema
        try:
            validated_data = schema.model_validate(result_data)
        except Exception as e:
            raise ExtractionError(f"Result doesn't match schema: {e}") from e

        # Build usage stats
        usage_stats: UsageStats | None = None
        if result_message and result_message.usage:
            usage = result_message.usage
            usage_stats = UsageStats(
                input_tokens=usage.get("input_tokens", 0),
                output_tokens=usage.get("output_tokens", 0),
                cache_read_input_tokens=usage.get("cache_read_input_tokens", 0),
                cache_creation_input_tokens=usage.get("cache_creation_input_tokens", 0),
            )

        return ExtractionResult(
            data=validated_data,
            model=model_used,
            duration_ms=result_message.duration_ms if result_message else 0,
            duration_api_ms=result_message.duration_api_ms if result_message else 0,
            num_turns=result_message.num_turns if result_message else 0,
            session_id=result_message.session_id if result_message else "",
            cost_usd=result_message.total_cost_usd if result_message else None,
            usage=usage_stats,
        )


__all__ = ["OpenXtract", "ExtractionResult", "UsageStats"]
