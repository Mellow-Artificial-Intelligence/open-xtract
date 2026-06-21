# openextract Roadmap

This roadmap is a planning document for turning `openextract` from a focused
alpha package into a stable, production-ready extraction library. It is meant to
be specific enough to turn into issues, milestones, and release gates, while
still leaving room to learn from users and provider behavior.

Planning horizon: 2026 H2 through 2028+.

Status baseline: `openextract` is currently an alpha Python package at `0.8.x`
with a compact public API, strict unit coverage, provider extras, a CLI, examples,
SSRF protections for URL input, pinned GitHub Actions, and a rolling dependency
embargo.

## North Star

`openextract` should be the simplest trustworthy way for Python developers to
turn unstructured media into validated Pydantic objects.

The library should feel:

- Small at the call site: bring a schema, choose a model, pass media, get a typed
  result.
- Provider-neutral: OpenAI, Anthropic, Google, Bedrock, xAI, local models, and
  other providers should fit the same user model wherever possible.
- Safe by default: remote inputs, dependencies, secrets, and logs should be
  handled conservatively.
- Honest about quality: extraction accuracy, cost, latency, and provider
  capability should be measured rather than implied.
- Stable for integrators: 1.0 should mean that the public API, error taxonomy,
  CLI contracts, and documentation can be depended on.

## Roadmap Principles

1. Correctness and reliability come before feature breadth.
2. The default API should stay boring and hard to misuse.
3. New complexity should earn its place with clear user value.
4. Provider-specific behavior should be isolated and documented.
5. Security-sensitive behavior should be opt-out, not opt-in.
6. Examples, docs, and error messages are part of the product surface.
7. Evaluation and benchmarks should guide claims about "best" models or flows.
8. The package should remain useful as a library, CLI, and building block for
   larger systems.

## Current Product Surface

### Public API

Already available:

- `extract`
- `extract_async`
- `extract_many`
- `extract_many_async`
- `extract_with_usage`
- `extract_with_usage_async`
- `Usage`
- `ExtractionError`
- `UrlFetchError`
- `SchemaValidationError`
- `ModelError`
- `ProviderNotInstalledError`

Current input forms:

- Local paths.
- `http://` and `https://` URLs.
- Raw `bytes` with explicit `media_type`.
- Binary file-like objects with explicit `media_type`.

Current output contract:

- A Pydantic model instance for the simple APIs.
- `(model_instance, Usage)` for usage APIs.
- Lists of outputs or exceptions for batch APIs.
- JSON or repr output from the CLI.

### Current Strengths

- Small public API that is easy to explain.
- Provider support through `pydantic-ai` and optional extras.
- Explicit exception hierarchy.
- CLI with structured exit codes.
- Async and batch support.
- Token usage surface.
- Strict unit coverage with a 100 percent coverage gate.
- URL fetch timeout and redirect limit.
- SSRF host validation for URL inputs and redirects.
- PEP 561 marker via `py.typed`.
- Pinned GitHub Actions.
- Rolling 24-hour dependency embargo through `uv`.
- Runnable examples and smoke tests.
- Basic benchmark script for local overhead.

### Current Constraints And Risks

- The package is marked alpha, so users may not know what is stable.
- The API is intentionally small, but several features are encoded as keyword
  arguments rather than a larger configuration object.
- `extract_many` uses `asyncio.run`, which is fine from sync code but needs clear
  behavior when called from an already-running event loop.
- URL handling is safer than a naive fetch, but remote input remains a
  security-sensitive boundary.
- Local files and URLs are fully read into memory before model submission.
- There is no formal provider capability matrix for supported media types,
  maximum sizes, usage reporting reliability, timeout behavior, or model quirks.
- There is no dedicated evaluation harness for extraction accuracy.
- Documentation is currently README-heavy plus a static landing page.
- Live provider testing is optional and limited.
- Batch behavior is useful but still basic: no rate limiting, checkpoints,
  per-item usage aggregation, or JSONL output.

## Strategic Workstreams

The roadmap is organized around nine workstreams. Milestones should pull from
multiple workstreams, because releases should improve the whole user experience
instead of shipping isolated features.

1. API stability and ergonomics.
2. Input and media handling.
3. Provider support and capability mapping.
4. Reliability, retries, timeouts, and batch scale.
5. Evaluation, benchmarks, and extraction quality.
6. Security, privacy, and supply-chain posture.
7. CLI and automation workflows.
8. Documentation, examples, and community.
9. Release operations and governance.

## Milestone Overview

| Horizon | Target | Theme | Primary Outcome |
| --- | --- | --- | --- |
| 2026 Q3 | 0.9 beta | Stabilize alpha | Public contracts are audited, documented, and hardened. |
| 2026 Q4 | 1.0 | Stable foundation | Core API, CLI, docs, provider matrix, and release process are production-ready. |
| 2027 H1 | 1.x | Production scale | Batch extraction, observability, rate limits, and large-input workflows mature. |
| 2027 H2 | 1.x / 2.0 planning | Quality layer | Evaluations, evidence, provenance, and repair loops become first-class. |
| 2028+ | 2.x | Ecosystem | Integrations, plugins, enterprise controls, and workflow orchestration expand. |

Dates are planning windows, not promises. The 1.0 release should be gated by
quality and contract readiness rather than a calendar date.

## Phase 0: Roadmap Governance

Target window: immediate.

Goal: make the roadmap actionable and keep it connected to releases.

### Deliverables

- Keep this file as the source of truth for long-range direction.
- Create GitHub milestones matching the major phases.
- Convert the immediate issue candidates near the end of this file into tracked
  issues.
- Add labels for workstreams:
  - `area:api`
  - `area:cli`
  - `area:docs`
  - `area:evaluation`
  - `area:inputs`
  - `area:providers`
  - `area:reliability`
  - `area:security`
  - `area:release`
- Add labels for urgency:
  - `priority:p0`
  - `priority:p1`
  - `priority:p2`
  - `priority:p3`
- Add labels for maturity:
  - `kind:bug`
  - `kind:feature`
  - `kind:docs`
  - `kind:refactor`
  - `kind:research`
  - `kind:test`

### Acceptance Criteria

- Every roadmap item selected for a release has an owner, issue, acceptance
  criteria, and test/doc expectations.
- Release notes cite completed roadmap items when relevant.
- Deferred items are explicitly moved, not silently forgotten.

## Phase 1: Stabilize Alpha Into Beta

Target window: 2026 Q3.

Recommended release target: `0.9.0`.

Theme: reduce ambiguity, harden boundaries, and document current behavior before
adding major new concepts.

### 1. Public API Audit

Purpose: decide what can become stable in 1.0 and what should remain
experimental.

Tasks:

- Audit every exported symbol in `openextract.__all__`.
- Document whether each exported symbol is stable, provisional, or internal.
- Confirm public function signatures are the right shape for 1.0.
- Decide whether retry and backoff options should remain separate keyword args
  or eventually live in a config object.
- Decide whether usage APIs should stay as separate functions or converge on a
  richer result object in a future release.
- Decide whether `extract_many` should remain sync-over-async or gain a clearer
  guard for active event loops.
- Add a compatibility policy covering breaking changes, deprecations, and
  minimum Python versions.

Acceptance criteria:

- README or docs include an API stability section.
- `CHANGELOG.md` calls out any planned deprecations before 1.0.
- No public API behavior changes without tests.

### 2. Error Model Hardening

Purpose: make failure modes precise and actionable without leaking provider
internals unnecessarily.

Tasks:

- Review current exception mapping for local file failures, URL failures,
  provider failures, validation failures, import failures, and unknown failures.
- Decide whether local file read failures deserve a dedicated `InputFileError`
  or should remain `ExtractionError`.
- Validate `max_retries` and `retry_backoff` inputs explicitly.
- Confirm retry behavior for negative values, zero values, and non-model errors.
- Add tests for retry validation and any new exception types.
- Improve CLI stderr wording where users need a clear next action.
- Document the full exception hierarchy with examples.

Acceptance criteria:

- All public exceptions have docs and at least one test proving when they occur.
- CLI exit codes remain documented and tested.
- Unexpected exceptions preserve useful context without exposing secrets.

### 3. Input Boundary Hardening

Purpose: keep the simple input API while making remote and large inputs safer.

Tasks:

- Add documented size expectations for local files, bytes, file-like inputs, and
  URLs.
- Consider a configurable maximum input size for URL responses.
- Consider a configurable maximum input size for local reads.
- Decide how to handle missing `Content-Length`, incorrect `Content-Length`, and
  compressed responses.
- Add explicit tests for oversized URL responses if size limits are added.
- Review URL scheme handling and confirm only intended schemes are accepted.
- Revisit DNS rebinding risk and document the current protection boundary.
- Keep `OPENEXTRACT_ALLOW_PRIVATE_URLS` documented as a deliberate opt-out.

Acceptance criteria:

- Users can understand what happens with large files before they hit memory or
  provider limits.
- Remote input behavior has tests for redirects, timeouts, blocked hosts,
  oversized content, and HTTP errors.
- Security docs mention URL input risk and safe defaults.

### 4. Provider Install And Capability Clarity

Purpose: make provider setup predictable and reduce support issues.

Tasks:

- Finish and release `ProviderNotInstalledError` and CLI exit code `6`.
- Verify every provider prefix maps to the right optional extra or install hint.
- Add docs for provider-specific environment variables.
- Add a provider capability matrix:
  - Text input.
  - Image input.
  - PDF input.
  - Audio input.
  - Video input.
  - Structured output reliability.
  - Usage reporting availability.
  - Known model identifier examples.
  - Required extra.
- Mark capability entries as verified, expected, or unknown.
- Avoid claiming support that is only theoretical.

Acceptance criteria:

- A user can install the right extra from the model string alone.
- Provider docs separate verified behavior from inherited `pydantic-ai` support.
- Unsupported or unknown behavior is documented plainly.

### 5. CLI Polishing

Purpose: make the CLI reliable enough for scripting and early production use.

Tasks:

- Finish and release `--continue-on-error`.
- Add examples for single-file JSON, batch JSON, partial failure, usage, stdin,
  and explicit media type.
- Consider JSONL output for large batches.
- Consider `--output pretty-json` versus compact JSON if script users need
  stable machine output.
- Add documented shell examples for capturing stderr and exit codes.
- Decide whether CLI should accept a config file before 1.0 or defer it.
- Ensure batch output remains stable enough to document as a contract.

Acceptance criteria:

- Every CLI flag has a README/docs example.
- Every exit code has a test.
- Batch partial failure shape is documented and tested.

### 6. Documentation Restructure

Purpose: make docs usable beyond the README.

Tasks:

- Split the README into:
  - quick start,
  - installation,
  - core API,
  - CLI,
  - examples,
  - troubleshooting,
  - security notes,
  - provider setup,
  - roadmap link.
- Decide whether static HTML remains enough or whether to adopt a docs generator
  before 1.0.
- Add an API reference page generated or maintained from docstrings.
- Add a troubleshooting guide for provider SDK imports, credentials, validation
  errors, URL fetch failures, and batch failures.
- Add a "choosing a model" page based on verified capability data.
- Keep examples synchronized with docs.

Acceptance criteria:

- New users can complete a first extraction, a CLI extraction, and a batch
  extraction from docs alone.
- Common failures have documented explanations and next steps.
- Docs do not require reading tests to understand behavior.

### 7. Beta Release Gate

Purpose: know when `0.9.0` is ready.

`0.9.0` should not ship until:

- Public API audit is complete.
- Error model behavior is documented.
- Provider install hints are released.
- CLI partial failure behavior is released or consciously deferred.
- Roadmap is linked from the README.
- Security docs cover URL input behavior.
- Targeted tests, lint, formatting check, and type check pass.

## Phase 2: 1.0 Stable Foundation

Target window: 2026 Q4.

Recommended release target: `1.0.0`.

Theme: establish stable contracts for the core library, CLI, docs, releases, and
security posture.

### 1. Define The 1.0 Contract

Purpose: make promises that integrators can rely on.

Stable in 1.0:

- Core function names and return shapes.
- Exception hierarchy and CLI exit codes.
- Basic input forms.
- Provider extra naming convention.
- Batch result ordering.
- `Usage` fields.
- Python version support.

Potentially still provisional:

- Provider capability claims for less-tested media types.
- Any future rich result object.
- Any advanced evaluation APIs.
- Any plugin or preprocessing extension API.

Tasks:

- Add a "Stability and compatibility" docs page.
- Add deprecation rules:
  - Deprecate in one minor release.
  - Remove only in the next major release unless security requires faster action.
  - Use clear changelog entries.
- Add a policy for provider behavior changes caused by upstream SDKs or
  `pydantic-ai`.
- Confirm minimum Python version and support window.

Acceptance criteria:

- Users can tell which APIs are stable.
- Breaking changes are not hidden in patch releases.
- Public behavior has regression tests.

### 2. Provider Compatibility Program

Purpose: keep "bring your own model" honest.

Tasks:

- Create a small live smoke suite that can run behind environment variables.
- Run live checks for at least three representative providers before 1.0:
  - one OpenAI-compatible provider,
  - one non-OpenAI cloud provider,
  - one local or local-compatible provider if practical.
- Record tested model IDs, media inputs, and observed usage behavior.
- Add a provider capability page to docs.
- Add issue templates for provider compatibility reports.
- Track upstream `pydantic-ai` breaking changes.

Acceptance criteria:

- 1.0 release notes include the provider set actually verified for release.
- Capability docs distinguish "verified" from "supported by dependency".
- Provider setup errors have actionable messages.

### 3. Result And Diagnostics Design

Purpose: decide how to expose richer metadata without making `extract()` heavy.

Research questions:

- Should the library add `extract_result()` or `extract_detailed()` returning an
  `ExtractResult[T]`?
- Should usage, model identifier, media type, input source, provider, timing,
  retries, and warnings live in one result object?
- Should batch APIs support per-item usage and diagnostics?
- How should diagnostics avoid logging secrets or sensitive input?
- Can the simple `extract()` API remain unchanged?

Likely shape:

- Keep `extract()` returning `T`.
- Keep `extract_with_usage()` for the common accounting case.
- Add a future richer API only if it solves evidence, diagnostics, and batch
  accounting cleanly.

Acceptance criteria:

- A design note exists before implementation.
- The simple API remains the recommended first path.
- Any new result object has tests for serialization and privacy-sensitive fields.

### 4. Batch Reliability Baseline

Purpose: make batch extraction safe enough for real workloads.

Tasks:

- Document `extract_many` concurrency behavior and ordering.
- Add explicit validation for `max_concurrency`.
- Decide behavior for `extract_many` called from an active event loop.
- Consider per-item timeout support.
- Consider per-item usage support.
- Consider retry summaries for batch outputs.
- Add tests for empty input, invalid concurrency, partial failure, and retries in
  batch mode.
- Add CLI docs for partial failure handling.

Acceptance criteria:

- Batch APIs have clear concurrency and error behavior.
- Batch CLI output can be consumed by scripts without guesswork.
- Invalid batch options fail early with actionable errors.

### 5. Release And Supply-Chain Readiness

Purpose: make releases repeatable and auditable.

Tasks:

- Add a release checklist.
- Confirm PyPI trusted publishing is configured correctly.
- Keep all GitHub Actions pinned to commit SHAs.
- Add a process for periodically refreshing pinned actions.
- Keep the dependency embargo workflow documented and healthy.
- Consider adding provenance or artifact attestation if it fits the release
  pipeline.
- Consider generating an SBOM if users request it.
- Add a security advisory process for GitHub Security Advisories.

Acceptance criteria:

- A maintainer can cut a release from docs without tribal knowledge.
- Release artifacts, changelog, tags, and PyPI versions agree.
- Security response expectations are documented.

### 6. 1.0 Release Gate

`1.0.0` should not ship until:

- Public API contract is documented.
- No known P0 or P1 correctness bugs remain open.
- CLI contracts are documented and tested.
- Provider capability matrix exists.
- At least three representative providers have passing live smoke checks or a
  documented reason they cannot be run.
- Security docs cover remote input, credentials, dependency embargo, and
  vulnerability reporting.
- Full local quality commands pass:
  - `uv run ruff check .`
  - `uv run ruff format --check .`
  - `uv run ty check`
  - `uv run pytest -v --cov=openextract --cov-report=term-missing`
  - `uv run coverage report --fail-under=100`
- README, docs, examples, changelog, and release notes are synchronized.

## Phase 3: Production Scale And Workflow Ergonomics

Target window: 2027 H1.

Theme: make `openextract` comfortable for sustained production usage, especially
batch jobs, larger media, and observability.

### 1. Large Input Strategy

Purpose: avoid forcing users to solve large documents and long media themselves.

Candidate capabilities:

- Document page selection.
- Document chunking.
- Multi-page extraction strategy.
- Image resizing or normalization as an optional preprocessing step.
- Audio segmentation strategy.
- Video frame or clip extraction strategy.
- Configurable maximum bytes before refusing or chunking.
- Clear provider-specific limits in docs.

Design constraints:

- Avoid heavy mandatory dependencies.
- Prefer optional extras for preprocessing.
- Keep raw-pass-through behavior available.
- Make preprocessing explicit so users understand cost and fidelity tradeoffs.

Acceptance criteria:

- Large input behavior is deterministic and documented.
- Optional preprocessing dependencies do not affect base install size.
- Tests cover chunk planning without requiring live model calls.

### 2. Rate Limiting, Backpressure, And Resumability

Purpose: support real batch jobs without overwhelming providers or losing all
progress on failure.

Candidate capabilities:

- Rate limit configuration by requests per minute.
- Token or byte budget hints where providers expose enough data.
- Per-item timeout.
- Per-item retry summary.
- Checkpoint output for CLI batch jobs.
- Resume failed batch items from a previous run.
- JSONL output for streaming batch results.
- Progress reporting to stderr for CLI jobs.

Acceptance criteria:

- Batch jobs can fail partially without losing successful results.
- Users can cap concurrency and request rate separately.
- CLI progress does not corrupt machine-readable stdout.

### 3. Observability And Cost Accounting

Purpose: make production behavior inspectable without forcing a logging stack.

Candidate capabilities:

- Per-call timing metadata.
- Retry count metadata.
- Provider/model metadata.
- Batch usage aggregation.
- Optional structured logging hooks.
- Optional Logfire integration docs through the existing `logfire` extra.
- Redaction rules for logs and diagnostics.

Acceptance criteria:

- Users can track cost and latency for a batch.
- Diagnostics do not include raw document bytes or secrets.
- Observability remains optional.

### 4. CLI As Automation Surface

Purpose: make the CLI useful in scripts, CI jobs, cron jobs, and data pipelines.

Candidate capabilities:

- JSONL output.
- Compact JSON mode.
- Output file support.
- Failure report file support.
- Config file support.
- Environment variable documentation for all config.
- Shell completion if maintenance cost is low.
- Stable examples for common shell workflows.

Acceptance criteria:

- CLI can process many inputs without holding all output until the end.
- stdout and stderr contracts are clear.
- CLI examples cover success, partial failure, and retry workflows.

### 5. Performance Baselines

Purpose: keep local overhead small and prevent regressions.

Tasks:

- Turn `scripts/bench.py` into a documented maintainer tool.
- Capture baseline import time, media-read overhead, agent construction cost,
  and mocked extraction overhead.
- Decide whether any benchmark should run in CI with generous thresholds.
- Add performance notes for users processing large batches.

Acceptance criteria:

- Maintainers can detect large local-overhead regressions.
- Performance docs separate local overhead from model/network latency.

## Phase 4: Extraction Quality, Evidence, And Human Review

Target window: 2027 H2.

Theme: move beyond "call a model" into measurable extraction quality,
traceability, and repair workflows.

### 1. Evaluation Harness

Purpose: measure extraction accuracy across schemas, media types, and models.

Candidate capabilities:

- Fixture datasets for invoices, receipts, document summaries, and forms.
- Golden expected outputs.
- Exact-match and field-level metrics.
- Type-aware comparison for numbers, dates, enums, lists, and optional fields.
- Semantic comparison only where exact comparison is inappropriate.
- Cost and latency capture.
- Provider/model comparison reports.
- Regression thresholds for selected fixtures.

Acceptance criteria:

- Extraction quality can be compared across model versions.
- Release notes can mention quality changes with evidence.
- Evaluation fixtures avoid sensitive or copyrighted private data.

### 2. Evidence And Provenance

Purpose: help users trust extracted values.

Candidate capabilities:

- Optional evidence fields with source snippets.
- Page, frame, timestamp, or byte-range references where available.
- Confidence or uncertainty fields, clearly caveated.
- Output warnings for missing, inferred, or conflicting values.
- A richer result object carrying output plus provenance.

Design constraints:

- Do not force evidence into the user's schema unless they ask for it.
- Do not claim precise provenance when the provider cannot supply it.
- Keep evidence snippets bounded to avoid exposing more sensitive content than
  necessary.

Acceptance criteria:

- Users can ask "where did this value come from?" for supported workflows.
- Evidence behavior is documented per media type and provider capability.
- Provenance metadata has privacy-conscious defaults.

### 3. Validation And Repair Loops

Purpose: improve extraction reliability when the first model response fails or
is incomplete.

Candidate capabilities:

- Schema-aware repair attempt after validation failure.
- Field-level retry for missing or invalid fields.
- User-provided validation hooks.
- Optional strict mode that refuses inferred values.
- Optional review mode that surfaces low-confidence fields.

Acceptance criteria:

- Repair behavior is explicit and testable.
- Validation hooks do not leak test-only logic into production paths.
- Users can choose between fast single-pass extraction and slower repair flows.

### 4. Domain Recipes

Purpose: provide high-quality examples without turning the core library into a
domain-specific product.

Candidate recipe areas:

- Invoices.
- Receipts.
- Bank statements.
- Contracts.
- Insurance forms.
- Meeting notes.
- Research papers.
- Product catalogs.

Guidelines:

- Recipes should live in docs or examples unless there is a strong reason to
  package them.
- Domain schemas should be examples, not hidden business rules.
- Regulated domains should include clear limitations.

Acceptance criteria:

- Recipes are runnable.
- Recipes include realistic failure handling.
- Recipes do not expand base dependencies unnecessarily.

## Phase 5: Ecosystem, Extensions, And Enterprise Readiness

Target window: 2028+.

Theme: let `openextract` fit into larger systems without sacrificing its simple
core.

### 1. Extension Points

Purpose: support advanced workflows while preserving the core API.

Candidate extension points:

- Input resolvers for custom storage backends.
- Preprocessors for media normalization.
- Postprocessors for validation, repair, and enrichment.
- Provider capability registry.
- Evaluation dataset plugins.
- Output sinks for JSONL, databases, queues, or object storage.

Acceptance criteria:

- Extension interfaces are small and stable.
- Core APIs remain usable without plugins.
- Plugins cannot silently weaken security defaults.

### 2. Workflow Integrations

Purpose: make `openextract` usable in common data and automation stacks.

Candidate integrations:

- Airflow examples.
- Dagster examples.
- Prefect examples.
- Celery/RQ examples.
- FastAPI service wrapper example.
- LangChain/LlamaIndex interoperability notes if users request them.
- Cloud storage examples for S3, GCS, and Azure Blob through user-provided file
  objects or optional resolvers.

Acceptance criteria:

- Integrations are examples first, dependencies second.
- Integration docs show retries, failure handling, and secret management.

### 3. Enterprise Controls

Purpose: support organizations with stricter compliance and governance needs.

Candidate capabilities:

- PII redaction hooks.
- Audit-friendly diagnostics.
- Strict no-remote-fetch mode.
- Allowlist-based URL fetching.
- Policy object for permitted providers and media types.
- Self-hosted/local model documentation.
- Secure logging guidance.

Acceptance criteria:

- Security controls are explicit and testable.
- Enterprise features do not make small-user workflows harder.
- Privacy-sensitive behavior is documented with examples.

### 4. Long-Term Maintenance

Purpose: keep the project healthy beyond feature work.

Tasks:

- Define supported Python versions by date and release line.
- Define supported `pydantic-ai` version ranges.
- Decide whether any `1.x` line needs long-term security support.
- Maintain issue triage cadence.
- Maintain provider compatibility reports.
- Keep examples from rotting as model names and providers change.

Acceptance criteria:

- Users can tell whether their version is supported.
- Security fixes have a predictable release path.
- Stale provider examples are removed or updated promptly.

## Cross-Workstream Backlog

This section lists detailed epics that can be scheduled into phases.

### API And Types

- Add API stability docs.
- Add explicit option validation for retry and concurrency settings.
- Decide whether to add `ExtractOptions`.
- Design `ExtractResult[T]` for future diagnostics, usage, evidence, and timing.
- Keep `extract()` returning only `T`.
- Add more type-check-focused tests for generic return behavior.
- Verify compatibility with common static type checkers where practical.
- Document sync versus async usage patterns.
- Clarify `BinaryIO` expectations and file-like object lifetime.

### Inputs And Media

- Add documented input size guidance.
- Add configurable max size for URL and local inputs if supported.
- Add optional content hashing for diagnostics and caching without exposing raw
  data.
- Add page-range support for documents if preprocessing is introduced.
- Add explicit media type override examples.
- Add MIME sniffing only if it can be done without surprising users or adding
  heavy dependencies.
- Add optional preprocessing extras for PDFs, images, audio, and video only when
  the dependency cost is justified.

### Providers

- Maintain provider prefix to extra mapping.
- Maintain provider environment variable docs.
- Maintain provider capability matrix.
- Add live smoke tests behind explicit environment variables.
- Track provider-specific structured-output quirks.
- Track provider-specific media limits.
- Track usage-reporting availability.
- Avoid provider-specific branches in core unless required for correctness.

### Reliability

- Validate retry settings.
- Validate concurrency settings.
- Add per-call timeout strategy if supported cleanly.
- Add rate limiting separate from concurrency.
- Add batch checkpointing and resume.
- Add JSONL streaming output.
- Add retry summaries.
- Add cancellation guidance for async users.
- Keep transient model failures distinct from validation failures.

### Evaluation

- Build fixture datasets with non-sensitive sample files.
- Define field-level comparison metrics.
- Capture cost and latency during live evals.
- Track quality regressions by release.
- Publish model comparison docs only when data is fresh and reproducible.
- Keep evaluation optional so normal tests stay fast and deterministic.

### Security And Privacy

- Keep SSRF protections on by default.
- Document private URL opt-out risks.
- Consider allowlist-based URL fetching for locked-down environments.
- Add response size limits.
- Avoid logging raw media, schema contents that may include secrets, or provider
  credentials.
- Keep GitHub Actions pinned.
- Maintain dependency embargo.
- Add security advisory workflow docs.
- Consider fuzzing URL parsing and input classification.

### CLI

- Stabilize exit codes.
- Add compact JSON and JSONL output if needed.
- Add progress to stderr only.
- Add output file support if it does not complicate stdout contracts.
- Add failure report support for batch jobs.
- Add config file support only after CLI flags settle.
- Add examples for common shell usage.

### Docs And Examples

- Keep README short enough for quick start.
- Move deeper material into docs pages.
- Add API reference.
- Add troubleshooting guide.
- Add provider setup guide.
- Add security guide.
- Add batch guide.
- Add evaluation guide once eval tooling exists.
- Keep examples runnable in smoke tests.
- Add migration guides for breaking releases.

### Release Operations

- Add release checklist.
- Keep changelog current.
- Keep release workflow minimal and auditable.
- Maintain pinned GitHub Actions.
- Maintain dependency embargo workflow.
- Ensure docs deploy after release.
- Ensure PyPI metadata stays accurate.
- Automate checks where automation reduces release risk.

## Suggested Issue Backlog

These are concrete issue candidates for the next planning pass.

### P0 Before 0.9

1. Audit public API and mark stable versus provisional.
   - Area: API.
   - Acceptance: docs list each exported symbol and expected stability.

2. Document compatibility and deprecation policy.
   - Area: release.
   - Acceptance: policy is linked from README or docs.

3. Validate retry and concurrency options.
   - Area: reliability.
   - Acceptance: invalid values fail early and tests cover edge cases.

4. Finish provider install error release path.
   - Area: providers.
   - Acceptance: error, CLI exit code, README docs, and changelog agree.

5. Document URL input security model.
   - Area: security.
   - Acceptance: docs cover SSRF protections, redirects, timeout, private URL
     opt-out, and remaining risk.

6. Add provider capability matrix.
   - Area: providers.
   - Acceptance: at least current providers are listed with verified, expected,
     or unknown capability status.

7. Link roadmap from README.
   - Area: docs.
   - Acceptance: README has a visible `Roadmap` link.

### P1 Before 1.0

8. Add API reference docs.
   - Area: docs.
   - Acceptance: public functions, parameters, returns, and errors are covered.

9. Add troubleshooting guide.
   - Area: docs.
   - Acceptance: provider installs, credentials, validation errors, URL fetch
     errors, and CLI failures are covered.

10. Add live provider smoke test harness.
    - Area: providers.
    - Acceptance: tests run only when explicit env vars and credentials are set.

11. Clarify sync batch behavior inside active event loops.
    - Area: API.
    - Acceptance: behavior is tested and documented.

12. Add input size limit design.
    - Area: inputs.
    - Acceptance: design covers URL, local path, bytes, file-like objects, and
      compatibility risks.

13. Add release checklist.
    - Area: release.
    - Acceptance: checklist covers version bump, changelog, docs, tests, tag,
      PyPI, GitHub release, and post-release validation.

14. Document CLI stdout and stderr contracts.
    - Area: CLI.
    - Acceptance: docs explain machine-readable stdout and human/progress stderr.

15. Turn benchmark script into documented maintainer tool.
    - Area: reliability.
    - Acceptance: docs explain what it measures and what it does not measure.

### P2 After 1.0

16. Add JSONL output for CLI batch.
    - Area: CLI.
    - Acceptance: output streams per item and preserves error records.

17. Add batch usage aggregation.
    - Area: reliability.
    - Acceptance: per-item and total usage are available without breaking simple
      output.

18. Design rate limiting separate from concurrency.
    - Area: reliability.
    - Acceptance: design handles provider quotas and tests timing behavior with
      mocked clocks where practical.

19. Build evaluation fixture dataset.
    - Area: evaluation.
    - Acceptance: includes at least invoices, receipts, document summary, and
      meeting-notes style fixtures with expected outputs.

20. Add field-level evaluation metrics.
    - Area: evaluation.
    - Acceptance: reports exact, normalized, missing, extra, and type-invalid
      fields.

21. Design evidence/provenance result shape.
    - Area: API.
    - Acceptance: design covers source snippets, page/timestamp references,
      confidence caveats, privacy, and backward compatibility.

22. Add optional large-document chunk planning.
    - Area: inputs.
    - Acceptance: chunk planning is testable without live model calls.

23. Add batch checkpoint and resume design.
    - Area: CLI.
    - Acceptance: design covers successful items, failed items, input identity,
      and repeatability.

### P3 Long Horizon

24. Add optional preprocessing extras.
    - Area: inputs.
    - Acceptance: extras do not affect base install and are documented clearly.

25. Add workflow integration examples.
    - Area: docs.
    - Acceptance: at least one orchestrator or service wrapper example includes
      retries, errors, and secret handling.

26. Add policy controls for providers and URL fetching.
    - Area: security.
    - Acceptance: users can restrict provider prefixes, remote URLs, and private
      network access explicitly.

27. Add plugin/extension design.
    - Area: API.
    - Acceptance: extension points are narrow, documented, and do not weaken
      default safety.

28. Add long-term support policy.
    - Area: release.
    - Acceptance: supported versions, Python versions, and security-fix policy
      are documented.

## Decision Records To Write

Before larger changes, write short design notes for:

- Public API stability and 1.0 contract.
- Rich result object versus separate helper functions.
- Input size limits and large-media handling.
- Provider capability matrix methodology.
- Batch rate limiting and resumability.
- Evidence/provenance metadata.
- Evaluation metrics and fixture policy.
- CLI JSONL and output contract.

Each design note should answer:

- What user problem is this solving?
- What is the smallest useful version?
- What alternatives were considered?
- What public contract does it create?
- What tests and docs are required?
- What could make this unsafe or hard to maintain?

## Metrics To Track

### Quality

- Unit test coverage remains at 100 percent unless the policy is intentionally
  changed.
- Type check passes on `src/openextract`.
- Live provider smoke tests pass for the providers selected for a release.
- Evaluation fixture quality does not regress beyond defined thresholds.

### Reliability

- Import time and local overhead remain within documented baselines.
- Batch partial failure behavior remains deterministic.
- Retry behavior is covered for sync, async, and batch paths.
- CLI exit codes remain stable.

### Security

- No known high-severity unpatched vulnerabilities in supported release lines.
- GitHub Actions remain pinned.
- Dependency embargo remains current.
- URL input protections remain covered by tests.
- Sensitive input and credentials are not logged by default.

### Usability

- A new user can complete a Python extraction from docs in under ten minutes.
- A new user can complete a CLI extraction from docs in under ten minutes.
- Provider setup failures point to the correct install extra.
- Common errors have troubleshooting entries.

## What Not To Do Yet

These are tempting but should wait until the foundations are stronger:

- Do not add a heavyweight preprocessing stack to the base install.
- Do not promise universal support for all media across all providers.
- Do not turn examples into production domain packages until there is user pull.
- Do not add a plugin system before the core extension points are clear.
- Do not publish model recommendations without fresh evaluation data.
- Do not add broad abstractions only to reduce a few keyword arguments.
- Do not weaken URL safety for convenience.

## Planning Cadence

Recommended cadence:

- Monthly: triage issues against this roadmap.
- Before every minor release: select a small coherent slice from the roadmap.
- Before every major release: review public contracts, deprecations, and migration
  notes.
- Quarterly: refresh provider capability data and example model identifiers.
- After every security-sensitive change: update `SECURITY.md`, tests, and docs.

## Near-Term Recommended Sequence

The next coherent sequence is:

1. Link this roadmap from the README.
2. Ship the unreleased provider install and CLI partial-failure work.
3. Add API stability and compatibility docs.
4. Add provider capability matrix.
5. Add URL/input security docs.
6. Validate retry and concurrency options.
7. Add CLI contract docs.
8. Cut `0.9.0` as beta.
9. Run live provider smoke checks.
10. Complete 1.0 release checklist and cut `1.0.0` only when release gates pass.

