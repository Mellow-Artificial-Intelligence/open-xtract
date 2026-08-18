"""Tests for openextract._swarm."""

import asyncio

import pytest
from pydantic import BaseModel
from pydantic_ai.models.test import TestModel

from openextract import (
    DefinedAgent,
    ExtractionInput,
    ExtractionResult,
    ModelError,
    SwarmMember,
    SwarmReduce,
    define_agent,
    extract_swarm,
    extract_swarm_async,
    extract_swarm_with_results,
    extract_swarm_with_results_async,
    resolve_swarm_members,
)
from openextract._swarm import _agent_instructions


class Person(BaseModel):
    name: str | None = None
    age: int | None = None
    tags: list[str] = []


class FakeUsage:
    input_tokens = 2
    output_tokens = 3
    total_tokens = 5


class FakeResult:
    def __init__(self, output):
        self.output = output

    @property
    def usage(self):
        return FakeUsage()


class FakeAgent:
    def __init__(self, schema, model, instructions):
        self.schema = schema
        self.model = model
        self.instructions = instructions


def install_agents(mocker, outcomes: dict[str, list]) -> list[FakeAgent]:
    """Replace agent construction and model calls with scripted outcomes.

    ``outcomes`` maps a model identifier to the queue of results (a dict of
    schema fields) or exceptions its calls produce, in call order. The returned
    list records every agent the swarm built.
    """
    built: list[FakeAgent] = []
    queues = {model: list(items) for model, items in outcomes.items()}

    def build(schema, model, instructions, **kwargs):
        agent = FakeAgent(schema, model, instructions)
        built.append(agent)
        return agent

    async def run(agent, inputs):
        outcome = queues[agent.model].pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return FakeResult(agent.schema.model_validate(outcome))

    mocker.patch("openextract._swarm._build_agent", side_effect=build)
    mocker.patch("openextract._swarm._run_extraction_async", side_effect=run)
    return built


class TestResolveSwarmMembers:
    def test_single_model_defaults_to_one_agent(self):
        assert resolve_swarm_members("test:a") == [SwarmMember("test:a")]

    def test_single_model_fans_out_to_size(self):
        members = resolve_swarm_members("test:a", 3)
        assert members == [SwarmMember("test:a")] * 3

    def test_model_list_becomes_one_member_each(self):
        members = resolve_swarm_members(["test:a", "test:b"])
        assert [member.model for member in members] == ["test:a", "test:b"]

    def test_tuples_are_accepted(self):
        assert len(resolve_swarm_members(("test:a", "test:b"))) == 2

    def test_swarm_members_pass_through(self):
        member = SwarmMember("test:a", instructions="focus", style="direct")
        assert resolve_swarm_members([member]) == [member]

    def test_a_configured_model_object_is_a_single_agent(self):
        model = TestModel()
        assert resolve_swarm_members(model) == [SwarmMember(model)]

    def test_empty_list_is_rejected(self):
        with pytest.raises(ValueError, match="at least one model"):
            resolve_swarm_members([])

    def test_an_agent_that_contributes_no_model_is_rejected(self):
        empty = DefinedAgent(description="group with nothing in it")
        with pytest.raises(ValueError, match="at least one model"):
            resolve_swarm_members(empty)

    def test_defined_agents_are_flattened_into_members(self):
        agent = define_agent(
            "Invoices",
            model="test:a",
            subagents=[define_agent("Totals", model="test:b")],
        )
        assert [member.model for member in resolve_swarm_members(agent)] == [
            "test:a",
            "test:b",
        ]

    def test_size_may_not_contradict_an_agent_list(self):
        with pytest.raises(ValueError, match="size cannot be combined"):
            resolve_swarm_members(["test:a", "test:b"], 3)

    def test_size_matching_the_agent_list_is_allowed(self):
        assert len(resolve_swarm_members(["test:a", "test:b"], 2)) == 2

    @pytest.mark.parametrize("size", [0, -1, 17, True, 2.0])
    def test_out_of_range_sizes_are_rejected(self, size):
        with pytest.raises(ValueError, match="size must be an integer from 1 to 16"):
            resolve_swarm_members("test:a", size)

    def test_agent_lists_are_bounded_too(self):
        with pytest.raises(ValueError, match="size must be an integer from 1 to 16"):
            resolve_swarm_members([f"test:{index}" for index in range(17)])


class TestAgentInstructions:
    def test_a_lone_agent_keeps_the_caller_instructions(self):
        assert _agent_instructions("focus", 0, 1) == "focus"

    def test_a_lone_agent_without_instructions_gets_none(self):
        assert _agent_instructions(None, 0, 1) is None

    def test_peers_get_an_independence_role(self):
        text = _agent_instructions(None, 1, 3)
        assert text is not None
        assert "agent 2 of 3" in text

    def test_caller_instructions_come_before_the_role(self):
        text = _agent_instructions("focus", 0, 2)
        assert text is not None
        assert text.startswith("focus\n\n")

    def test_blank_caller_instructions_are_dropped(self):
        text = _agent_instructions("   ", 0, 2)
        assert text is not None
        assert text.startswith("You are extraction agent")


class TestExtractSwarm:
    def test_reduced_output_merges_every_agent(self, mocker):
        install_agents(
            mocker,
            {
                "test:a": [{"name": "Ada", "tags": ["x"]}],
                "test:b": [{"age": 36, "tags": ["y"]}],
            },
        )
        result = extract_swarm(Person, ["test:a", "test:b"], b"doc", media_type="text/plain")
        assert result == Person(name="Ada", age=36, tags=["x", "y"])

    async def test_async_sibling_returns_the_same_output(self, mocker):
        install_agents(mocker, {"test:a": [{"name": "Ada"}, {"name": "Ada"}]})
        result = await extract_swarm_async(
            Person, "test:a", b"doc", media_type="text/plain", size=2
        )
        assert result == Person(name="Ada")

    def test_reduce_first_keeps_the_leading_agent(self, mocker):
        install_agents(mocker, {"test:a": [{"name": "Ada"}], "test:b": [{"name": "Grace"}]})
        result = extract_swarm(
            Person, ["test:a", "test:b"], b"doc", media_type="text/plain", reduce="first"
        )
        assert result == Person(name="Ada")

    def test_reduce_vote_keeps_the_majority(self, mocker):
        install_agents(
            mocker,
            {
                "test:a": [{"name": "Ada"}],
                "test:b": [{"name": "Grace"}],
                "test:c": [{"name": "Grace"}],
            },
        )
        result = extract_swarm(
            Person,
            ["test:a", "test:b", "test:c"],
            b"doc",
            media_type="text/plain",
            reduce=SwarmReduce.VOTE,
        )
        assert result == Person(name="Grace")

    def test_sync_entry_points_refuse_a_running_loop(self, mocker):
        install_agents(mocker, {"test:a": [{"name": "Ada"}]})

        async def call():
            with pytest.raises(RuntimeError, match="extract_swarm"):
                extract_swarm(Person, "test:a", b"doc", media_type="text/plain")

        asyncio.run(call())

    def test_structured_inputs_are_accepted(self, mocker):
        install_agents(mocker, {"test:a": [{"name": "Ada"}]})
        item = ExtractionInput(b"doc", media_type="text/plain", name="doc.txt")
        assert extract_swarm(Person, "test:a", item) == Person(name="Ada")


class TestExtractSwarmWithResults:
    def test_per_agent_diagnostics_are_reported(self, mocker):
        install_agents(mocker, {"test:a": [{"name": "Ada"}], "test:b": [{"age": 36}]})
        swarm = extract_swarm_with_results(
            Person, ["test:a", "test:b"], b"doc", media_type="text/plain"
        )
        assert swarm.output == Person(name="Ada", age=36)
        assert swarm.reduce is SwarmReduce.MERGE
        assert swarm.usage.total_tokens == 10
        assert [agent.model for agent in swarm.agents] == ["test:a", "test:b"]
        assert all(agent.attempts == 1 and agent.duration >= 0 for agent in swarm.agents)
        assert {agent.media_type for agent in swarm.agents} == {"text/plain"}

    async def test_async_sibling_reports_the_same_diagnostics(self, mocker):
        install_agents(mocker, {"test:a": [{"name": "Ada"}]})
        swarm = await extract_swarm_with_results_async(
            Person, "test:a", b"doc", media_type="text/plain"
        )
        assert swarm.output == Person(name="Ada")
        assert len(swarm.agents) == 1

    def test_source_labels_come_from_the_input(self, mocker, tmp_path):
        install_agents(mocker, {"test:a": [{"name": "Ada"}]})
        local = tmp_path / "input.txt"
        local.write_bytes(b"hello")
        swarm = extract_swarm_with_results(Person, "test:a", local)
        assert swarm.agents[0].source == "path 'input.txt'"

    def test_progress_callbacks_fire_per_agent(self, mocker):
        install_agents(mocker, {"test:a": [{"name": "Ada"}], "test:b": [{"age": 36}]})
        started: list[tuple[int, int]] = []
        finished: list[tuple[int, int, object]] = []
        swarm = extract_swarm_with_results(
            Person,
            ["test:a", "test:b"],
            b"doc",
            media_type="text/plain",
            max_concurrency=1,
            on_agent_start=lambda index, total: started.append((index, total)),
            on_agent=lambda index, total, result: finished.append((index, total, result)),
        )
        assert started == [(0, 2), (1, 2)]
        assert [(index, total) for index, total, _ in finished] == [(0, 2), (1, 2)]
        assert [result for _, _, result in finished] == list(swarm.agents)

    def test_a_failed_agent_is_reported_without_losing_the_swarm(self, mocker):
        failure = ModelError("provider down", retryable=False)
        install_agents(mocker, {"test:a": [failure], "test:b": [{"name": "Grace"}]})
        swarm = extract_swarm_with_results(
            Person, ["test:a", "test:b"], b"doc", media_type="text/plain"
        )
        assert swarm.agents[0] is failure
        assert isinstance(swarm.agents[1], ExtractionResult)
        assert swarm.output == Person(name="Grace")
        assert swarm.usage.total_tokens == 5

    def test_every_agent_failing_raises_the_first_failure(self, mocker):
        first = ModelError("first", retryable=False)
        install_agents(
            mocker,
            {"test:a": [first], "test:b": [ModelError("second", retryable=False)]},
        )
        with pytest.raises(ModelError, match="first"):
            extract_swarm_with_results(
                Person, ["test:a", "test:b"], b"doc", media_type="text/plain"
            )

    def test_agents_retry_transient_model_errors(self, mocker):
        install_agents(mocker, {"test:a": [ModelError("flaky"), {"name": "Ada"}]})
        swarm = extract_swarm_with_results(
            Person,
            "test:a",
            b"doc",
            media_type="text/plain",
            max_retries=1,
            retry_backoff=0,
        )
        assert swarm.agents[0].attempts == 2
        assert swarm.output == Person(name="Ada")


class TestSwarmConfiguration:
    def test_the_input_is_loaded_once_for_the_whole_swarm(self, mocker):
        install_agents(mocker, {"test:a": [{"name": "Ada"}] * 3})
        load = mocker.patch(
            "openextract._swarm._get_media_async",
            return_value=(b"doc", "text/plain"),
        )
        extract_swarm(Person, "test:a", b"doc", media_type="text/plain", size=3)
        assert load.call_count == 1

    def test_swarm_instructions_reach_every_agent(self, mocker):
        built = install_agents(mocker, {"test:a": [{"name": "Ada"}], "test:b": [{"name": "Ada"}]})
        extract_swarm(Person, ["test:a", "test:b"], b"doc", "focus", media_type="text/plain")
        assert all(agent.instructions.startswith("focus\n\n") for agent in built)
        assert {agent.instructions for agent in built} == {
            _agent_instructions("focus", 0, 2),
            _agent_instructions("focus", 1, 2),
        }

    def test_member_overrides_beat_the_swarm_wide_values(self, mocker):
        built = install_agents(mocker, {"test:a": [{"name": "Ada"}], "test:b": [{"name": "Ada"}]})
        extract_swarm(
            Person,
            [SwarmMember("test:a", instructions="only tables", style="direct"), "test:b"],
            b"doc",
            "focus",
            media_type="text/plain",
        )
        by_model = {agent.model: agent.instructions for agent in built}
        assert by_model["test:a"].startswith("only tables\n\n")
        assert by_model["test:b"].startswith("focus\n\n")

    def test_a_lone_agent_is_not_told_about_peers(self, mocker):
        built = install_agents(mocker, {"test:a": [{"name": "Ada"}]})
        extract_swarm(Person, "test:a", b"doc", "focus", media_type="text/plain")
        assert built[0].instructions == "focus"

    def test_invalid_member_styles_fail_before_any_model_call(self, mocker):
        install_agents(mocker, {"test:a": [{"name": "Ada"}]})
        with pytest.raises(ValueError, match="style must be one of"):
            extract_swarm(
                Person,
                [SwarmMember("test:a", style="sideways")],
                b"doc",
                media_type="text/plain",
            )

    @pytest.mark.parametrize("max_concurrency", [0, -1, True])
    def test_invalid_concurrency_is_rejected(self, max_concurrency):
        with pytest.raises(ValueError, match="max_concurrency must be a positive integer"):
            extract_swarm(
                Person,
                "test:a",
                b"doc",
                media_type="text/plain",
                max_concurrency=max_concurrency,
            )

    def test_invalid_retry_options_are_rejected(self):
        with pytest.raises(ValueError, match="max_retries must be a non-negative integer"):
            extract_swarm(Person, "test:a", b"doc", media_type="text/plain", max_retries=-1)

    def test_concurrency_never_exceeds_the_agent_count(self, mocker):
        install_agents(mocker, {"test:a": [{"name": "Ada"}] * 2})
        swarm = extract_swarm_with_results(
            Person, "test:a", b"doc", media_type="text/plain", size=2, max_concurrency=8
        )
        assert len(swarm.agents) == 2
