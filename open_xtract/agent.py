import json
import re
from bisect import bisect_right

from typing_extensions import Any

from agents import Agent, FunctionTool, RunContextWrapper, function_tool


@function_tool
def grep(
    ctx: RunContextWrapper[Any],
    pattern: str,
    text: str,
    ignore_case: bool = False,
    multiline: bool = False,
    dotall: bool = False,
    context_lines: int = 0,
    max_matches: int | None = None,
    use_regex: bool = True,
) -> str:
    """Search text with an advanced grep-like matcher and return formatted results.

    Args:
        pattern: Regex (or literal) pattern to search for.
        text: The input text to search.
        ignore_case: Case-insensitive matching.
        multiline: ^ and $ match start/end of each line.
        dotall: Dot matches newline.
        context_lines: Number of lines to show before/after each match.
        max_matches: Max number of matches to return; unlimited if None.
        use_regex: If False, treat the pattern as a literal string.
    """
    flags = 0
    if ignore_case:
        flags |= re.IGNORECASE
    if multiline:
        flags |= re.MULTILINE
    if dotall:
        flags |= re.DOTALL

    compiled = re.compile(re.escape(pattern) if not use_regex else pattern, flags)

    # Precompute line boundaries for fast line-number mapping
    lines_with_endings = text.splitlines(True)
    starts: list[int] = []
    bounds: list[tuple[int, int]] = []
    pos = 0
    for chunk in lines_with_endings:
        start = pos
        end = start + len(chunk)
        starts.append(start)
        bounds.append((start, end))
        pos = end

    def line_index_for(position: int) -> int:
        if not starts:
            return 0
        i = bisect_right(starts, position) - 1
        return max(0, i)

    output_lines: list[str] = []
    match_count = 0

    for m in compiled.finditer(text):
        if max_matches is not None and match_count >= max_matches:
            break
        line_idx = line_index_for(m.start())
        # Determine context range
        start_ctx = max(0, line_idx - context_lines)
        end_ctx = min(len(bounds) - 1, line_idx + context_lines)

        # Emit before context
        for i in range(start_ctx, line_idx):
            line_text = text[bounds[i][0]:bounds[i][1]].rstrip("\n")
            output_lines.append(f"- {i+1}:{line_text}")

        # Emit match line
        match_line = text[bounds[line_idx][0]:bounds[line_idx][1]].rstrip("\n")
        output_lines.append(f": {line_idx+1}:{match_line}")

        # Emit after context
        for i in range(line_idx + 1, end_ctx + 1):
            line_text = text[bounds[i][0]:bounds[i][1]].rstrip("\n")
            output_lines.append(f"- {i+1}:{line_text}")

        match_count += 1

    if match_count == 0:
        return ""

    return "\n".join(output_lines)


agent = Agent(
    name="GrepAgent",
    tools=[grep],
)

for tool in agent.tools:
    if isinstance(tool, FunctionTool):
        print(tool.name)
        print(tool.description)
        print(json.dumps(tool.params_json_schema, indent=2))
        print()