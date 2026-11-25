"""Guardrailed LangGraph Ofwat PR24 & Environment Act 2021 compliance copilot."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
from langgraph.graph import END, START, StateGraph
from pypdf import PdfReader
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass(frozen=True)
class RegulatoryDocument:
    """Authoritative UK water regulatory clause for hybrid retrieval and citation verification."""

    doc_id: str
    framework: str
    section: str
    title: str
    text: str
    keywords: tuple[str, ...]


REGULATORY_CORPUS: tuple[RegulatoryDocument, ...] = (
    RegulatoryDocument(
        doc_id="OFWAT-PR24-ODI-01",
        framework="Ofwat PR24 Final Determinations",
        section="AMP8 Storm Overflow Performance Commitment",
        title="Storm Overflow Spill Count ODI & Underperformance Penalties",
        text=(
            "Under Ofwat PR24 Final Determinations (2025-2030 AMP8), water and sewerage companies "
            "face Outcome Delivery Incentive (ODI) penalties for storm overflow spills exceeding "
            "company-specific annual targets (benchmark <= 20 spills per overflow per year toward "
            "the 40% reduction milestone by 2030). Standard ODI underperformance rates apply per "
            "excess spill and escalate for repeated failures and Price Control Deliverables (PCDs)."
        ),
        keywords=("ofwat", "pr24", "odi", "penalty", "storm overflow", "amp8", "spill count"),
    ),
    RegulatoryDocument(
        doc_id="OFWAT-PR24-PCD-02",
        framework="Ofwat PR24 Final Determinations",
        section="Price Control Deliverable Clawback & Pollution Incidents",
        title="Serious Pollution Incidents & Capital Enhancement Clawback",
        text=(
            "Ofwat PR24 imposes severe ODI penalties for Category 1-3 pollution incidents and "
            "requires automatic customer bill clawbacks where storm overflow enhancement schemes "
            "fail to deliver contracted storage or green-infrastructure spill reductions."
        ),
        keywords=("pr24", "clawback", "pcd", "pollution incident", "enhancement"),
    ),
    RegulatoryDocument(
        doc_id="ENV-ACT-2021-S141A-01",
        framework="Environment Act 2021",
        section="Section 141A Storm Overflow Discharge Reduction Plan",
        title="Statutory Duty to Progressively Reduce Harm from Storm Overflows",
        text=(
            "Section 141A of the Environment Act 2021 places a statutory duty on sewerage "
            "undertakers to eliminate ecological harm and reduce adverse impacts on public health "
            "from storm overflows: 75% reduction of discharges into high-priority sites (including "
            "designated Bathing Waters and chalk streams) by 2035, and 100% across all sites by 2050."
        ),
        keywords=("environment act 2021", "section 141a", "141a", "bathing water", "chalk stream", "2035", "2050"),
    ),
    RegulatoryDocument(
        doc_id="ENV-ACT-2021-S141B-02",
        framework="Environment Act 2021",
        section="Section 141B & 141C Near-Real-Time Spill & Water Quality Reporting",
        title="1-Hour Public EDM Disclosure & Downstream Water Quality Monitoring",
        text=(
            "Sections 141B-141C require sewerage undertakers to publish Event Duration Monitoring "
            "(EDM) spill start and stop notifications within 1 hour of discharge onset and maintain "
            "continuous upstream/downstream dissolved oxygen, turbidity, and ammonia monitoring."
        ),
        keywords=("141b", "141c", "1-hour", "real-time", "dissolved oxygen", "turbidity", "monitoring"),
    ),
    RegulatoryDocument(
        doc_id="EA-EDM-PERMIT-01",
        framework="Environment Agency EDM Permitting Rules",
        section="Dry-Weather Overflow Prohibition & Permit Breach Enforcement",
        title="Prohibition of Dry-Weather Spills (Dry-Day Spilling)",
        text=(
            "Environment Agency environmental permits strictly prohibit Combined Sewer Overflow "
            "(CSO) discharges during dry weather. A spill occurring when 24-hour antecedent rainfall "
            "is <= 0.25 mm and Antecedent Precipitation Index (API) is <= 2.0 mm constitutes a "
            "suspected dry-day spill permit breach subject to EA investigation and unlimited fines."
        ),
        keywords=("ea", "edm", "dry-weather", "dry-day", "permit", "breach", "rainfall"),
    ),
    RegulatoryDocument(
        doc_id="EA-EDM-DATA-02",
        framework="Environment Agency EDM Permitting Rules",
        section="15-Minute Telemetry Completeness & Flatlined Sensor Quarantine",
        title="EDM Uptime >= 90% / 98% and 12/24-Hour Counting Methodology",
        text=(
            "EA EDM guidance requires >= 90% telemetry operational completeness (rising to 98% under "
            "PR24 best practice). Stuck or flatlined level/DO sensors must be quarantined immediately "
            "and replaced by validated spatial interpolation or physical inspection records."
        ),
        keywords=("edm", "uptime", "completeness", "flatlined", "sensor", "quarantine", "12/24"),
    ),
)


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _infer_framework_from_filename(path: Path) -> str:
    stem = path.stem.lower()
    if "ofwat" in stem or "pr24" in stem:
        return "Ofwat PR24 Final Determinations"
    if "environment" in stem or "141a" in stem or "act" in stem:
        return "Environment Act 2021"
    if "edm" in stem or "ea" in stem or "permit" in stem:
        return "Environment Agency EDM Permitting Rules"
    return f"UK Water Statutory Document ({path.name})"


def _parse_markdown_sections(raw_text: str, source_path: Path) -> list[RegulatoryDocument]:
    framework = _infer_framework_from_filename(source_path)
    docs: list[RegulatoryDocument] = []
    heading_pattern = re.compile(r"^##\s+(?:\[([A-Z0-9\-]+)\]\s+)?(.+)$", re.MULTILINE)
    matches = list(heading_pattern.finditer(raw_text))

    if not matches:
        clean_text = " ".join(raw_text.split())
        if clean_text:
            stem_id = re.sub(r"[^A-Z0-9]+", "-", source_path.stem.upper()).strip("-")
            docs.append(
                RegulatoryDocument(
                    doc_id=f"DISK-{stem_id}-01",
                    framework=framework,
                    section=source_path.stem,
                    title=source_path.stem.replace("_", " ").title(),
                    text=clean_text,
                    keywords=tuple(_tokenize(clean_text)[:12]),
                )
            )
        return docs

    stem_id = re.sub(r"[^A-Z0-9]+", "-", source_path.stem.upper()).strip("-")
    for idx, match in enumerate(matches):
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(raw_text)
        body = " ".join(raw_text[start:end].split())
        if not body:
            continue
        doc_id = match.group(1) or f"DISK-{stem_id}-{idx + 1:02d}"
        title = match.group(2).strip()
        docs.append(
            RegulatoryDocument(
                doc_id=doc_id,
                framework=framework,
                section=title,
                title=title,
                text=body,
                keywords=tuple(_tokenize(f"{title} {body}")[:12]),
            )
        )
    return docs


def _extract_pdf_text(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages).strip()


def load_ofwat_regulatory_corpus(
    docs_dir: Path = Path("data/regulatory_docs"),
    include_builtin: bool = True,
) -> tuple[RegulatoryDocument, ...]:
    """Load `.pdf`, `.md`, and `.txt` regulatory docs from disk and merge with statutory corpus."""
    by_id: dict[str, RegulatoryDocument] = {}
    if include_builtin:
        for doc in REGULATORY_CORPUS:
            by_id[doc.doc_id] = doc

    dir_path = Path(docs_dir)
    if dir_path.exists() and dir_path.is_dir():
        supported_files = sorted(
            p
            for p in dir_path.iterdir()
            if p.is_file() and p.suffix.lower() in {".md", ".txt", ".pdf"}
        )
        for file_path in supported_files:
            suffix = file_path.suffix.lower()
            if suffix == ".pdf":
                raw_text = _extract_pdf_text(file_path)
                parsed_docs = _parse_markdown_sections(raw_text, file_path)
            else:
                raw_text = file_path.read_text(encoding="utf-8")
                parsed_docs = _parse_markdown_sections(raw_text, file_path)

            for doc in parsed_docs:
                by_id[doc.doc_id] = doc

    return tuple(by_id.values())


class HybridRegulatoryRetriever:
    """Hybrid BM25 + TF-IDF cosine retriever over UK water statutory documents."""

    def __init__(self, docs: tuple[RegulatoryDocument, ...] = REGULATORY_CORPUS) -> None:
        self.docs = docs
        corpus_texts = [f"{d.framework} {d.section} {d.title} {d.text}" for d in docs]
        self._tokenized = [_tokenize(t) for t in corpus_texts]
        self._bm25 = BM25Okapi(self._tokenized)
        self._tfidf = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        self._tfidf_matrix = self._tfidf.fit_transform(corpus_texts)

    def retrieve(self, query: str, top_k: int = 3) -> list[tuple[RegulatoryDocument, float]]:
        q_tokens = _tokenize(query)
        bm25_raw = np.asarray(self._bm25.get_scores(q_tokens), dtype=float)
        if bm25_raw.max() > 0:
            bm25_norm = bm25_raw / bm25_raw.max()
        else:
            bm25_norm = bm25_raw

        q_vec = self._tfidf.transform([query])
        tfidf_scores = cosine_similarity(q_vec, self._tfidf_matrix).ravel()
        if tfidf_scores.max() > 0:
            tfidf_norm = tfidf_scores / tfidf_scores.max()
        else:
            tfidf_norm = tfidf_scores

        combined = 0.55 * bm25_norm + 0.45 * tfidf_norm
        ranked_idx = np.argsort(combined)[::-1][:top_k]
        return [(self.docs[i], round(float(combined[i]), 4)) for i in ranked_idx]


def calculate_statutory_penalty_exposure(
    annual_spill_count: int = 0,
    total_spill_hours: float = 0.0,
    dry_day_spills: int = 0,
    sensitive_site: bool = False,
    edm_uptime_pct: float = 99.0,
    water_company_code: str = "THM",
) -> dict[str, Any]:
    """Deterministic statutory breach and Ofwat PR24 / EA penalty exposure calculator."""
    spill_threshold = 10 if sensitive_site else 20
    excess_spills = max(0, int(annual_spill_count) - spill_threshold)
    per_spill_rate = 11_000.0 if sensitive_site else 5_500.0
    odi_spill_penalty = excess_spills * per_spill_rate

    hour_threshold = 40.0 if sensitive_site else 80.0
    excess_hours = max(0.0, float(total_spill_hours) - hour_threshold)
    odi_hour_penalty = excess_hours * 1_250.0

    dry_spill_penalty = max(0, int(dry_day_spills)) * 25_000.0
    uptime_shortfall = max(0.0, 90.0 - float(edm_uptime_pct))
    edm_data_penalty = uptime_shortfall * 4_000.0

    breach_flags: list[str] = []
    if dry_day_spills > 0:
        breach_flags.append("EA_DRY_WEATHER_PERMIT_BREACH (EA-EDM-PERMIT-01)")
    if excess_spills > 0:
        breach_flags.append("OFWAT_PR24_ODI_SPILL_EXCEEDANCE (OFWAT-PR24-ODI-01)")
    if sensitive_site and annual_spill_count > 10:
        breach_flags.append("ENV_ACT_2021_S141A_HIGH_PRIORITY_RISK (ENV-ACT-2021-S141A-01)")
    if edm_uptime_pct < 90.0:
        breach_flags.append("EA_EDM_COMPLETENESS_SHORTFALL (EA-EDM-DATA-02)")

    total_exposure = odi_spill_penalty + odi_hour_penalty + dry_spill_penalty + edm_data_penalty

    if dry_day_spills > 0 or edm_uptime_pct < 90.0 or excess_spills >= 15:
        status = "STATUTORY_BREACH"
    elif excess_spills > 0 or excess_hours > 0:
        status = "AT_RISK"
    else:
        status = "COMPLIANT"

    return {
        "water_company_code": water_company_code,
        "compliance_status": status,
        "sensitive_site": sensitive_site,
        "spill_threshold": spill_threshold,
        "excess_spills": excess_spills,
        "odi_penalty_gbp": round(odi_spill_penalty + odi_hour_penalty, 2),
        "ea_enforcement_exposure_gbp": round(dry_spill_penalty + edm_data_penalty, 2),
        "total_exposure_gbp": round(total_exposure, 2),
        "breach_flags": breach_flags,
    }


class CopilotState(TypedDict):
    """LangGraph workflow state for Ofwat PR24 & Environment Act compliance copilot."""

    query: str
    telemetry_context: dict[str, Any]
    docs_dir: str | None
    guardrail_passed: bool
    guardrail_reason: str
    retrieved_docs: list[dict[str, Any]]
    draft_answer: str
    cited_doc_ids: list[str]
    citations_verified: bool
    verification_notes: str
    penalty_exposure: dict[str, Any]
    final_answer: str


_DEFAULT_CORPUS = load_ofwat_regulatory_corpus(Path("data/regulatory_docs"))
_RETRIEVER = HybridRegulatoryRetriever(_DEFAULT_CORPUS)
_VALID_DOC_IDS = {doc.doc_id for doc in _DEFAULT_CORPUS}
_BLOCKED_PATTERNS = re.compile(
    r"(ignore previous instructions|bypass|jailbreak|drop table|rm -rf|<script>)",
    re.IGNORECASE,
)


def _guardrail_node(state: CopilotState) -> dict[str, Any]:
    query = (state.get("query") or "").strip()
    if not query:
        return {
            "guardrail_passed": False,
            "guardrail_reason": "Empty regulatory query.",
        }
    if _BLOCKED_PATTERNS.search(query):
        return {
            "guardrail_passed": False,
            "guardrail_reason": "Query blocked by safety/prompt-injection guardrail.",
        }
    return {"guardrail_passed": True, "guardrail_reason": "ok"}


def _route_after_guardrail(state: CopilotState) -> str:
    return "retrieve" if state["guardrail_passed"] else "fallback"


def _retrieve_node(state: CopilotState) -> dict[str, Any]:
    docs_dir = state.get("docs_dir")
    retriever = (
        HybridRegulatoryRetriever(load_ofwat_regulatory_corpus(Path(docs_dir)))
        if docs_dir
        else _RETRIEVER
    )
    hits = retriever.retrieve(state["query"], top_k=3)
    docs_payload = [
        {
            "doc_id": doc.doc_id,
            "framework": doc.framework,
            "section": doc.section,
            "title": doc.title,
            "text": doc.text,
            "score": score,
        }
        for doc, score in hits
    ]
    return {"retrieved_docs": docs_payload}


def _draft_compliance_node(state: CopilotState) -> dict[str, Any]:
    docs = state.get("retrieved_docs", [])
    ctx = state.get("telemetry_context") or {}

    penalty = calculate_statutory_penalty_exposure(
        annual_spill_count=int(ctx.get("annual_spill_count", 0)),
        total_spill_hours=float(ctx.get("total_spill_hours", 0.0)),
        dry_day_spills=int(ctx.get("dry_day_spills", 0)),
        sensitive_site=bool(ctx.get("sensitive_site", False)),
        edm_uptime_pct=float(ctx.get("edm_uptime_pct", 99.0)),
        water_company_code=str(ctx.get("water_company_code", "THM")),
    )

    relevant_docs = [d for d in docs if d["score"] > 0.05] or docs[:2]
    cited_ids = [d["doc_id"] for d in relevant_docs]

    clauses = "\n".join(
        f"- [{d['doc_id']}] {d['framework']} ({d['section']}): {d['text']}"
        for d in relevant_docs
    )
    answer = (
        f"Regulatory Compliance Assessment ({penalty['water_company_code']} — "
        f"Status: {penalty['compliance_status']}):\n"
        f"{clauses}\n"
        f"Deterministic Exposure Estimate: £{penalty['total_exposure_gbp']:,.2f} "
        f"(Ofwat PR24 ODI: £{penalty['odi_penalty_gbp']:,.2f}; "
        f"EA Enforcement/EDM: £{penalty['ea_enforcement_exposure_gbp']:,.2f})."
    )
    if penalty["breach_flags"]:
        answer += "\nTriggered Statutory Flags: " + ", ".join(penalty["breach_flags"])

    return {
        "draft_answer": answer,
        "cited_doc_ids": cited_ids,
        "penalty_exposure": penalty,
    }


def _citation_verifier_node(state: CopilotState) -> dict[str, Any]:
    """Judge node verifying every cited doc ID exists in retrieved context & statutory corpus."""
    draft = state.get("draft_answer", "")
    retrieved_ids = {d["doc_id"] for d in state.get("retrieved_docs", [])}
    cited_in_text = re.findall(r"\[([A-Z0-9\-]+)\]", draft)
    valid_ids = _VALID_DOC_IDS | retrieved_ids

    if not cited_in_text:
        return {
            "citations_verified": False,
            "verification_notes": "No inline statutory citations found in draft.",
        }

    invalid = [cid for cid in cited_in_text if cid not in valid_ids or cid not in retrieved_ids]
    if invalid:
        return {
            "citations_verified": False,
            "verification_notes": f"Unverified citations detected: {invalid}",
        }

    return {
        "citations_verified": True,
        "verification_notes": f"Verified {len(cited_in_text)} statutory citation(s).",
        "final_answer": draft,
    }


def _route_after_verifier(state: CopilotState) -> str:
    return "done" if state.get("citations_verified") else "fallback"


def _deterministic_fallback_node(state: CopilotState) -> dict[str, Any]:
    """Deterministic statutory breach/penalty calculator fallback when guardrails or citations fail."""
    if not state.get("guardrail_passed", True):
        return {
            "citations_verified": False,
            "penalty_exposure": {},
            "cited_doc_ids": [],
            "final_answer": (
                f"Request declined by compliance guardrail: {state.get('guardrail_reason', 'Invalid query')}. "
                "Please submit a valid Ofwat PR24, Environment Act 2021, or EA EDM compliance query."
            ),
        }

    ctx = state.get("telemetry_context") or {}
    penalty = state.get("penalty_exposure") or calculate_statutory_penalty_exposure(
        annual_spill_count=int(ctx.get("annual_spill_count", 0)),
        total_spill_hours=float(ctx.get("total_spill_hours", 0.0)),
        dry_day_spills=int(ctx.get("dry_day_spills", 0)),
        sensitive_site=bool(ctx.get("sensitive_site", False)),
        edm_uptime_pct=float(ctx.get("edm_uptime_pct", 99.0)),
        water_company_code=str(ctx.get("water_company_code", "THM")),
    )
    fallback_answer = (
        f"[Deterministic Statutory Fallback] Status: {penalty['compliance_status']} | "
        f"Total Exposure: £{penalty['total_exposure_gbp']:,.2f} | "
        f"Flags: {', '.join(penalty['breach_flags']) or 'None'}. "
        f"(Verification note: {state.get('verification_notes', 'fallback invoked')})"
    )
    return {
        "penalty_exposure": penalty,
        "final_answer": fallback_answer,
    }


def build_ofwat_copilot_graph():
    """Compile the guardrailed LangGraph compliance workflow."""
    graph = StateGraph(CopilotState)
    graph.add_node("guardrail", _guardrail_node)
    graph.add_node("retrieve", _retrieve_node)
    graph.add_node("draft", _draft_compliance_node)
    graph.add_node("verify_citations", _citation_verifier_node)
    graph.add_node("fallback", _deterministic_fallback_node)

    graph.add_edge(START, "guardrail")
    graph.add_conditional_edges(
        "guardrail",
        _route_after_guardrail,
        {"retrieve": "retrieve", "fallback": "fallback"},
    )
    graph.add_edge("retrieve", "draft")
    graph.add_edge("draft", "verify_citations")
    graph.add_conditional_edges(
        "verify_citations",
        _route_after_verifier,
        {"done": END, "fallback": "fallback"},
    )
    graph.add_edge("fallback", END)
    return graph.compile()


def run_ofwat_compliance_copilot(
    query: str,
    telemetry_context: dict[str, Any] | None = None,
    docs_dir: Path | str | None = None,
) -> CopilotState:
    """Execute the Ofwat PR24 / Environment Act 2021 compliance copilot."""
    app = build_ofwat_copilot_graph()
    initial_state: CopilotState = {
        "query": query,
        "telemetry_context": telemetry_context or {},
        "docs_dir": str(docs_dir) if docs_dir is not None else None,
        "guardrail_passed": True,
        "guardrail_reason": "",
        "retrieved_docs": [],
        "draft_answer": "",
        "cited_doc_ids": [],
        "citations_verified": False,
        "verification_notes": "",
        "penalty_exposure": {},
        "final_answer": "",
    }
    return app.invoke(initial_state)
