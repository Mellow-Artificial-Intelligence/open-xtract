"""Minimal Streamlit demo for running the OpenXtract PDF pipeline.

The app exposes a single PDF uploader and uses a curated Pydantic schema to
summarise business-style reports. The schema is fixed in code so that users can
quickly see the kind of structured output OpenXtract can produce without
configuring any fields in the UI.
"""

from __future__ import annotations

import inspect
import os
from typing import Any, List, Optional, Sequence, Union

import streamlit as st
from open_xtract import OpenXtract
from pydantic import BaseModel, Field

DEFAULT_MODEL = "openrouter:x-ai/grok-4-fast:free"


class KeyMetric(BaseModel):
    """Quantitative metric that the report highlights."""

    name: str = Field(..., description="Name of the metric or KPI.")
    value: str = Field(..., description="Reported value exactly as written.")
    unit: Optional[str] = Field(None, description="Unit of measure tied to the value.")
    trend: Optional[str] = Field(None, description="Direction or interpretation of change.")
    notes: Optional[str] = Field(None, description="Context that explains the metric.")


class ReportFinding(BaseModel):
    """Narrative insight or conclusion surfaced in the report."""

    title: str = Field(..., description="Headline for the finding.")
    impact: Optional[str] = Field(None, description="Impact or severity noted in the report.")
    evidence: Optional[str] = Field(
        None,
        description="Supporting evidence, figures, or citations backing the finding.",
    )


class Recommendation(BaseModel):
    """Action the report suggests taking."""

    title: str = Field(..., description="Short label for the recommendation.")
    rationale: Optional[str] = Field(None, description="Why this recommendation matters.")
    priority: Optional[str] = Field(None, description="Priority or urgency stated in the report.")
    owner: Optional[str] = Field(None, description="Suggested owner or responsible party.")
    due_date: Optional[str] = Field(None, description="Timeline or deadline if provided.")


class ReportSummary(BaseModel):
    """Structured summary for business or analytical reports."""

    report_title: str = Field(..., description="Title of the report.")
    report_date: Optional[str] = Field(None, description="Publication or delivery date.")
    organization: Optional[str] = Field(None, description="Company or client the report is for.")
    prepared_by: Optional[str] = Field(None, description="Person or team that authored the report.")
    report_type: Optional[str] = Field(None, description="Type of report (e.g. quarterly review, audit).")
    executive_summary: str = Field(..., description="Short narrative overview of the report.")
    key_metrics: List[KeyMetric] = Field(
        default_factory=list,
        description="Important quantitative metrics spotlighted in the report.",
    )
    key_findings: List[ReportFinding] = Field(
        default_factory=list,
        description="Primary findings or insights highlighted in the report.",
    )
    recommendations: List[Recommendation] = Field(
        default_factory=list,
        description="Actions suggested based on the findings.",
    )
    follow_up_actions: List[str] = Field(
        default_factory=list,
        description="Next steps, owners, or timelines mentioned for follow-up.",
    )


def _clean_text(value: Optional[str]) -> str:
    return value.strip() if isinstance(value, str) else ""


def _value_or_default(value: Optional[str], *, fallback: str) -> str:
    cleaned = _clean_text(value)
    return cleaned if cleaned else fallback


def _ensure_summary(data: Union[ReportSummary, dict[str, Any]]) -> ReportSummary:
    if isinstance(data, ReportSummary):
        return data
    return ReportSummary.model_validate(data)


def _render_section_heading(label: str) -> None:
    st.subheader(label)
    st.divider()


def _render_metrics(metrics: Sequence[KeyMetric]) -> None:
    if not metrics:
        st.write(
            "The document does not surface quantitative indicators. Capture at least one KPI in future revisions to support the narrative findings."
        )
        return

    for index, metric in enumerate(metrics, start=1):
        name = _value_or_default(metric.name, fallback=f"Key Metric {index}")
        value = _value_or_default(metric.value, fallback="Value not specified in the source document.")
        trend = _clean_text(metric.trend)
        notes = _clean_text(metric.notes)
        unit = _clean_text(metric.unit)

        metric_sentence = f"**{name}** — {value}"
        if unit:
            metric_sentence += f" ({unit})"

        st.markdown(metric_sentence)

        narrative_bits: List[str] = []
        if trend:
            narrative_bits.append(f"Observed trend: {trend}.")
        if notes:
            narrative_bits.append(notes)
        if not narrative_bits:
            narrative_bits.append(
                "No qualitative commentary was attached to this metric; validate its implications during stakeholder reviews."
            )
        st.caption(" ".join(narrative_bits))


def _render_findings(findings: Sequence[ReportFinding]) -> None:
    if not findings:
        st.write(
            "The analyser did not extract explicit findings. Treat this as a cue to formulate headline insights after reviewing the document manually."
        )
        return

    for index, finding in enumerate(findings, start=1):
        title = _value_or_default(finding.title, fallback=f"Finding {index}")
        impact = _clean_text(finding.impact)
        evidence = _clean_text(finding.evidence)

        headline = f"**{title}**"
        if impact:
            headline += f" — Impact assessed as {impact}."
        else:
            headline += " — Impact not explicitly rated; confirm with the authoring team."

        st.markdown(headline)
        st.caption(evidence if evidence else "Evidence reference not surfaced; corroborate during validation.")


def _render_recommendations(items: Sequence[Recommendation]) -> None:
    if not items:
        st.write(
            "No forward-looking actions were captured automatically. Propose next steps based on the dominant findings before circulating the deliverable."
        )
        return

    for index, recommendation in enumerate(items, start=1):
        title = _value_or_default(recommendation.title, fallback=f"Recommendation {index}")
        rationale = _value_or_default(
            recommendation.rationale,
            fallback="Provide supporting rationale before execution.",
        )
        priority = _value_or_default(
            recommendation.priority,
            fallback="Priority not ranked; assign urgency during the next review.",
        )
        owner = _value_or_default(
            recommendation.owner,
            fallback="No owner assigned; nominate accountable lead.",
        )
        due_date = _value_or_default(
            recommendation.due_date,
            fallback="Timeline to be confirmed.",
        )

        st.markdown(f"**{title}**")
        st.caption(f"Priority: {priority} | Owner: {owner} | Timeline: {due_date}")
        st.write(rationale)


def _render_follow_up(actions: Sequence[str]) -> None:
    if not actions:
        st.write(
            "Follow-up milestones were not explicitly identified. Schedule a checkpoint to confirm responsibilities and success criteria."
        )
        return

    for index, action in enumerate(actions, start=1):
        narrative = _value_or_default(
            action,
            fallback="Detail the action item before circulating this report further.",
        )
        st.write(f"Action {index}: {narrative}")


@st.cache_resource
def _get_client(model_name: str) -> OpenXtract:
    return OpenXtract(model=model_name)


def display_report(data: Union[ReportSummary, dict[str, Any]]) -> None:
    summary = _ensure_summary(data)

    st.success("Structured report extracted. Review the sections below.")

    _render_section_heading("Overview")

    report_title = _value_or_default(summary.report_title, fallback="Untitled report")
    organization = _value_or_default(
        summary.organization, fallback="an unspecified organisation"
    )
    report_type = _value_or_default(summary.report_type, fallback="a general report")
    report_date = _value_or_default(summary.report_date, fallback="an unspecified date")
    prepared_by = _value_or_default(
        summary.prepared_by, fallback="an unnamed authoring team"
    )

    st.write(
        f"{report_title} was prepared for {organization}, positioned as {report_type}, and delivered on {report_date} by {prepared_by}."
    )

    _render_section_heading("Executive Summary")
    st.write(
        _value_or_default(
            summary.executive_summary,
            fallback="The document did not provide an executive narrative; synthesise key storylines before publication.",
        )
    )

    _render_section_heading("Key Metrics")
    _render_metrics(summary.key_metrics)

    _render_section_heading("Key Findings")
    _render_findings(summary.key_findings)

    _render_section_heading("Recommendations")
    _render_recommendations(summary.recommendations)

    _render_section_heading("Follow-up Actions")
    _render_follow_up(summary.follow_up_actions)

    st.divider()
    with st.expander("Raw structured output", expanded=False):
        st.json(summary.model_dump())


def main() -> None:
    st.set_page_config(page_title="OpenXtract Report Reader", page_icon="📄")
    st.title("OpenXtract Report Reader")
    st.caption(
        "Upload a report-style PDF and generate an expert grade analysis."
    )

    if not os.environ.get("OPENROUTER_API_KEY"):
        st.info(
            "Set the `OPENROUTER_API_KEY` environment variable before using the extractor."
        )

    with st.expander("Show Pydantic schema used", expanded=False):
        schema_source = "\n\n".join(
            [
                inspect.getsource(KeyMetric),
                inspect.getsource(ReportFinding),
                inspect.getsource(Recommendation),
                inspect.getsource(ReportSummary),
            ]
        )
        st.code(schema_source, language="python")

    uploaded_pdf = None
    pdf_bytes: Optional[bytes] = None

    with st.form("openxtract-upload"):
        uploaded_pdf = st.file_uploader(
            "Upload a PDF",
            type=["pdf"],
            accept_multiple_files=False,
            help="Select the report you would like OpenXtract to analyse.",
        )

        if uploaded_pdf is not None:
            pdf_bytes = uploaded_pdf.getvalue()
            if not pdf_bytes:
                st.error("The uploaded file appears to be empty.")
            else:
                st.caption(f"Ready to analyse: {uploaded_pdf.name}")

        extract_clicked = st.form_submit_button("Extract insights")

    if not extract_clicked:
        return

    if not uploaded_pdf:
        st.warning("Upload a PDF before running extraction.")
        return

    if not pdf_bytes:
        st.error("The uploaded file appears to be empty.")
        return

    with st.spinner("Extracting structured report…"):
        try:
            client = _get_client(DEFAULT_MODEL)
            result = client.extract(pdf_bytes, ReportSummary)
        except Exception as exc:  # pragma: no cover - interactive feedback path
            st.error(f"Extraction failed: {exc}")
            return

    display_report(result)


if __name__ == "__main__":
    main()
