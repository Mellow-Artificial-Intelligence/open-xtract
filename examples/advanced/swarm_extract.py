"""Run several agents over one document and reduce their outputs.

Uses two models on the same file, then merges the results so a field either
model recovered ends up in the final object.
"""

from pydantic import BaseModel

from examples._shared import DOCUMENT_PAGE, anthropic_model, openai_model
from openextract import extract_swarm_with_results


class DocumentInfo(BaseModel):
    title: str
    summary: str
    topics: list[str] = []


def main() -> None:
    swarm = extract_swarm_with_results(
        schema=DocumentInfo,
        agents=[openai_model(), anthropic_model()],
        input_file=str(DOCUMENT_PAGE),
        instructions="Return a short title, a one-sentence summary, and the topics covered.",
        reduce="merge",
    )

    print(swarm.output.model_dump_json(indent=2))
    print()
    for index, agent in enumerate(swarm.agents):
        if isinstance(agent, Exception):
            print(f"agent {index}: failed with {type(agent).__name__}: {agent}")
        else:
            print(f"agent {index}: {agent.model} in {agent.duration:.2f}s")
    print(f"\nreduce={swarm.reduce} total_tokens={swarm.usage.total_tokens}")


if __name__ == "__main__":
    main()
