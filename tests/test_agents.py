"""Tests for openextract._agents and openextract.auth."""

import base64
import sys

import pytest
from pydantic import BaseModel

from openextract import (
    DefinedAgent,
    RemoteAgent,
    SwarmMember,
    define_agent,
    define_remote_agent,
    flatten_agent,
    load_agent,
    load_agent_directory,
    load_agents,
    resolve_output_schema,
)
from openextract._agents import is_agent, resolve_provided
from openextract.auth import basic, bearer, vercel_oidc


class Person(BaseModel):
    name: str


def write_agent(path, body: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return path


AGENT_SOURCE = """
from openextract import define_agent

agent = define_agent({description!r}, model={model!r})
"""


def agent_source(description: str, model: str = "test:a") -> str:
    return AGENT_SOURCE.format(description=description, model=model)


class TestDefineAgent:
    def test_builds_a_frozen_agent(self):
        agent = define_agent("Invoices", model="test:a", instructions="totals")
        assert isinstance(agent, DefinedAgent)
        assert (agent.description, agent.model, agent.instructions) == (
            "Invoices",
            "test:a",
            "totals",
        )

    def test_description_is_required(self):
        with pytest.raises(ValueError, match="description is required"):
            define_agent("   ", model="test:a")

    def test_non_string_descriptions_are_rejected(self):
        with pytest.raises(ValueError, match="description is required"):
            define_agent(None, model="test:a")

    def test_model_or_subagents_is_required(self):
        with pytest.raises(ValueError, match="requires model or subagents"):
            define_agent("Group")

    def test_subagents_alone_make_a_group(self):
        child = define_agent("Child", model="test:a")
        group = define_agent("Group", subagents=[child])
        assert group.model is None
        assert group.subagents == (child,)

    def test_output_schema_must_be_a_model(self):
        with pytest.raises(TypeError, match="BaseModel subclass"):
            define_agent("Invoices", model="test:a", output_schema={"type": "object"})

    def test_output_schema_is_kept(self):
        agent = define_agent("Invoices", model="test:a", output_schema=Person)
        assert resolve_output_schema(agent) is Person

    def test_a_missing_output_schema_is_reported_by_name(self):
        with pytest.raises(ValueError, match="'Invoices' is missing output_schema"):
            resolve_output_schema(define_agent("Invoices", model="test:a"))


class TestDefineRemoteAgent:
    def test_builds_a_remote_agent_with_defaults(self):
        agent = define_remote_agent("https://agents.example.com", "Invoices")
        assert isinstance(agent, RemoteAgent)
        assert agent.path == "/extract"

    def test_description_is_required(self):
        with pytest.raises(ValueError, match="description is required"):
            define_remote_agent("https://agents.example.com", "")

    def test_url_is_required(self):
        with pytest.raises(ValueError, match="url is required"):
            define_remote_agent("  ", "Invoices")

    def test_a_callable_url_is_accepted(self):
        agent = define_remote_agent(lambda: "https://agents.example.com", "Invoices")
        assert callable(agent.url)

    def test_output_schema_must_be_a_model(self):
        with pytest.raises(TypeError, match="BaseModel subclass"):
            define_remote_agent("https://agents.example.com", "Invoices", output_schema=int)


class TestIsAgent:
    def test_recognizes_defined_and_remote_agents(self):
        assert is_agent(define_agent("A", model="test:a"))
        assert is_agent(define_remote_agent("https://a.example.com", "A"))

    def test_rejects_plain_values(self):
        assert not is_agent("test:a")


class TestFlattenAgent:
    def test_a_model_id_becomes_one_member(self):
        assert flatten_agent("test:a") == [SwarmMember("test:a")]

    def test_a_swarm_member_passes_through(self):
        member = SwarmMember("test:a", instructions="focus")
        assert flatten_agent(member) == [member]

    def test_a_local_agent_carries_its_configuration(self):
        agent = define_agent("A", model="test:a", instructions="focus", style="direct")
        assert flatten_agent(agent) == [SwarmMember("test:a", "focus", "direct")]

    def test_a_remote_agent_becomes_a_remote_member(self):
        agent = define_remote_agent("https://a.example.com", "A")
        assert flatten_agent(agent) == [SwarmMember(agent)]

    def test_a_group_contributes_itself_then_its_children(self):
        child = define_agent("Child", model="test:b")
        group = define_agent("Group", model="test:a", subagents=[child])
        assert [member.model for member in flatten_agent(group)] == ["test:a", "test:b"]

    def test_a_pure_group_contributes_only_its_children(self):
        group = define_agent(
            "Group",
            subagents=[define_agent("Child", model="test:b"), "test:c"],
        )
        assert [member.model for member in flatten_agent(group)] == ["test:b", "test:c"]

    def test_nesting_is_depth_first(self):
        leaf = define_agent("Leaf", model="test:c")
        middle = define_agent("Middle", model="test:b", subagents=[leaf])
        root = define_agent("Root", model="test:a", subagents=[middle])
        assert [member.model for member in flatten_agent(root)] == ["test:a", "test:b", "test:c"]


class TestLoadAgent:
    def test_a_defined_agent_passes_through(self):
        agent = define_agent("A", model="test:a")
        assert load_agent(agent) is agent

    def test_loads_a_python_file(self, tmp_path):
        path = write_agent(tmp_path / "invoices.py", agent_source("Invoices"))
        assert load_agent(str(path)).description == "Invoices"

    def test_accepts_a_path_object(self, tmp_path):
        path = write_agent(tmp_path / "invoices.py", agent_source("Invoices"))
        assert load_agent(path).description == "Invoices"

    def test_a_file_without_an_agent_is_reported(self, tmp_path):
        path = write_agent(tmp_path / "empty.py", "value = 1\n")
        with pytest.raises(ValueError, match="must define an 'agent'"):
            load_agent(str(path))

    def test_a_failed_import_does_not_leave_a_module_behind(self, tmp_path):
        path = write_agent(tmp_path / "broken.py", "raise RuntimeError('boom')\n")
        before = set(sys.modules)
        with pytest.raises(RuntimeError, match="boom"):
            load_agent(str(path))
        assert set(sys.modules) == before

    def test_loads_a_module_attribute(self, monkeypatch, tmp_path):
        write_agent(tmp_path / "pkg_agent.py", agent_source("Invoices"))
        monkeypatch.syspath_prepend(str(tmp_path))
        assert load_agent("pkg_agent:agent").description == "Invoices"

    def test_a_module_attribute_that_is_not_an_agent_is_reported(self, monkeypatch, tmp_path):
        write_agent(tmp_path / "not_agent.py", "value = 1\n")
        monkeypatch.syspath_prepend(str(tmp_path))
        with pytest.raises(ValueError, match="is not a define_agent"):
            load_agent("not_agent:value")

    @pytest.mark.parametrize("spec", ["", "   ", None, 5])
    def test_unusable_specs_are_rejected(self, spec):
        with pytest.raises(ValueError, match="agent must be a define_agent value"):
            load_agent(spec)

    def test_a_bare_name_is_reported_as_a_bad_path(self):
        with pytest.raises(ValueError, match="Expected a directory, a Python file"):
            load_agent("no-such-agent")


class TestLoadAgents:
    def test_splits_a_comma_separated_string(self, tmp_path):
        first = write_agent(tmp_path / "a.py", agent_source("First"))
        second = write_agent(tmp_path / "b.py", agent_source("Second"))
        agents = load_agents(f"{first}, {second}")
        assert [agent.description for agent in agents] == ["First", "Second"]

    def test_accepts_a_sequence(self, tmp_path):
        agent = define_agent("A", model="test:a")
        assert load_agents([agent]) == [agent]

    @pytest.mark.parametrize("spec", ["", " , ", []])
    def test_an_empty_specification_is_rejected(self, spec):
        with pytest.raises(ValueError, match="at least one agent path"):
            load_agents(spec)


class TestLoadAgentDirectory:
    def test_loads_agent_py(self, tmp_path):
        write_agent(tmp_path / "invoices" / "agent.py", agent_source("Invoices"))
        assert load_agent(str(tmp_path / "invoices")).description == "Invoices"

    def test_instructions_md_fills_a_missing_instructions_field(self, tmp_path):
        root = tmp_path / "invoices"
        write_agent(root / "agent.py", agent_source("Invoices"))
        (root / "instructions.md").write_text("Totals only.\n", encoding="utf-8")
        assert load_agent_directory(root).instructions == "Totals only."

    def test_declared_instructions_win_over_instructions_md(self, tmp_path):
        root = tmp_path / "invoices"
        write_agent(
            root / "agent.py",
            "from openextract import define_agent\n"
            "agent = define_agent('Invoices', model='test:a', instructions='declared')\n",
        )
        (root / "instructions.md").write_text("from disk", encoding="utf-8")
        assert load_agent_directory(root).instructions == "declared"

    def test_an_empty_instructions_file_is_ignored(self, tmp_path):
        root = tmp_path / "invoices"
        write_agent(root / "agent.py", agent_source("Invoices"))
        (root / "instructions.md").write_text("   \n", encoding="utf-8")
        assert load_agent_directory(root).instructions is None

    def test_subagents_are_attached_in_sorted_order(self, tmp_path):
        root = tmp_path / "invoices"
        write_agent(root / "agent.py", agent_source("Invoices"))
        write_agent(root / "subagents" / "b_totals.py", agent_source("Totals"))
        write_agent(root / "subagents" / "a_lines.py", agent_source("Lines"))
        agent = load_agent_directory(root)
        assert [child.description for child in agent.subagents] == ["Lines", "Totals"]

    def test_nested_subagent_directories_are_loaded(self, tmp_path):
        root = tmp_path / "invoices"
        write_agent(root / "agent.py", agent_source("Invoices"))
        write_agent(root / "subagents" / "nested" / "agent.py", agent_source("Nested"))
        agent = load_agent_directory(root)
        assert [child.description for child in agent.subagents] == ["Nested"]

    def test_private_and_non_python_entries_are_skipped(self, tmp_path):
        root = tmp_path / "invoices"
        write_agent(root / "agent.py", agent_source("Invoices"))
        write_agent(root / "subagents" / "_helper.py", "value = 1\n")
        write_agent(root / "subagents" / ".hidden.py", "value = 1\n")
        write_agent(root / "subagents" / "notes.md", "text\n")
        assert load_agent_directory(root).subagents == ()

    def test_a_lone_subagent_becomes_the_directory_agent(self, tmp_path):
        root = tmp_path / "invoices"
        write_agent(root / "subagents" / "only.py", agent_source("Only"))
        assert load_agent_directory(root).description == "Only"

    def test_several_subagents_become_a_group_named_for_the_directory(self, tmp_path):
        root = tmp_path / "invoices"
        write_agent(root / "subagents" / "a.py", agent_source("Lines"))
        write_agent(root / "subagents" / "b.py", agent_source("Totals"))
        group = load_agent_directory(root)
        assert group.description == "invoices"
        assert group.model is None
        assert [child.description for child in group.subagents] == ["Lines", "Totals"]

    def test_a_group_directory_still_reads_instructions_md(self, tmp_path):
        root = tmp_path / "invoices"
        write_agent(root / "subagents" / "a.py", agent_source("Lines"))
        write_agent(root / "subagents" / "b.py", agent_source("Totals"))
        (root / "instructions.md").write_text("Group guidance.", encoding="utf-8")
        assert load_agent_directory(root).instructions == "Group guidance."

    def test_a_remote_agent_py_is_returned_unchanged(self, tmp_path):
        root = tmp_path / "remote"
        write_agent(
            root / "agent.py",
            "from openextract import define_remote_agent\n"
            "agent = define_remote_agent('https://a.example.com', 'Remote')\n",
        )
        write_agent(root / "subagents" / "a.py", agent_source("Ignored"))
        assert isinstance(load_agent_directory(root), RemoteAgent)

    def test_an_empty_directory_is_reported(self, tmp_path):
        root = tmp_path / "empty"
        root.mkdir()
        with pytest.raises(ValueError, match="No agent.py or subagents/ found"):
            load_agent_directory(root)


class TestResolveProvided:
    async def test_plain_values_pass_through(self):
        assert await resolve_provided("value") == "value"

    async def test_sync_providers_are_called(self):
        assert await resolve_provided(lambda: "value") == "value"

    async def test_async_providers_are_awaited(self):
        async def provide():
            return "value"

        assert await resolve_provided(provide) == "value"


class TestAuthHelpers:
    async def test_bearer_sends_a_static_token(self):
        assert await bearer("abc")() == {"Authorization": "Bearer abc"}

    async def test_bearer_resolves_a_provider_per_request(self):
        tokens = iter(["one", "two"])
        provide = bearer(lambda: next(tokens))
        assert await provide() == {"Authorization": "Bearer one"}
        assert await provide() == {"Authorization": "Bearer two"}

    async def test_basic_encodes_the_credentials(self):
        headers = await basic(("ada", "s3cret"))()
        encoded = base64.b64encode(b"ada:s3cret").decode()
        assert headers == {"Authorization": f"Basic {encoded}"}

    async def test_basic_accepts_a_provider(self):
        headers = await basic(lambda: ("ada", "s3cret"))()
        assert headers["Authorization"].startswith("Basic ")

    async def test_vercel_oidc_reads_the_token_per_request(self, monkeypatch):
        monkeypatch.setenv("VERCEL_OIDC_TOKEN", "token-1")
        provide = vercel_oidc()
        assert await provide() == {"Authorization": "Bearer token-1"}
        monkeypatch.setenv("VERCEL_OIDC_TOKEN", "token-2")
        assert await provide() == {"Authorization": "Bearer token-2"}

    async def test_vercel_oidc_reports_a_missing_token(self, monkeypatch):
        monkeypatch.delenv("VERCEL_OIDC_TOKEN", raising=False)
        with pytest.raises(ValueError, match="VERCEL_OIDC_TOKEN is not set"):
            await vercel_oidc()()
