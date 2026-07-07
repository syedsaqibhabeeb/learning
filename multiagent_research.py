"""
Multi-Agent Research Report API
--------------------------------

Run:
    pip install fastapi uvicorn pydantic openai tavily-python
    export OPENAI_API_KEY="your-openai-key"        # optional but recommended
    export TAVILY_API_KEY="your-tavily-key"        # optional but recommended
    export OPENAI_MODEL="gpt-5.5"                 # optional
    uvicorn main:app --reload

Test:
    curl -X POST "http://127.0.0.1:8000/research" \
      -H "Content-Type: application/json" \
      -d '{"topic":"How agentic AI will change cybersecurity operations","max_sources":5}'
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import textwrap
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator


# Optional dependencies.
# The app still runs without these, but quality improves when keys are present.
try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None

try:
    from tavily import TavilyClient
except Exception:  # pragma: no cover
    TavilyClient = None


# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger("multi_agent_research")


# ---------------------------------------------------------------------
# API Models
# ---------------------------------------------------------------------

class ResearchDepth(str, Enum):
    basic = "basic"
    standard = "standard"
    deep = "deep"


class ResearchRequest(BaseModel):
    topic: str = Field(
        ...,
        min_length=3,
        max_length=300,
        description="Topic to research.",
    )
    max_sources: int = Field(
        default=5,
        ge=2,
        le=10,
        description="Number of sources to retrieve.",
    )
    depth: ResearchDepth = Field(
        default=ResearchDepth.standard,
        description="Controls search depth and report detail.",
    )

    @field_validator("topic")
    @classmethod
    def clean_topic(cls, value: str) -> str:
        value = " ".join(value.strip().split())
        if not value:
            raise ValueError("Topic cannot be empty.")
        return value


class SourceDocument(BaseModel):
    source_id: str
    title: str
    url: str
    content: str
    score: float = 0.0
    provider: str = "unknown"


class ResearchSummary(BaseModel):
    short_summary: str
    key_findings: List[str]
    important_evidence: List[str]
    open_questions: List[str]
    source_coverage: str


class StructuredReport(BaseModel):
    executive_summary: str
    findings: List[str]
    risks: List[str]
    recommendations: List[str]
    limitations: List[str]


class AgentTraceEvent(BaseModel):
    agent: str
    status: str
    message: str
    timestamp: str


class ResearchResponse(BaseModel):
    topic: str
    created_at: str
    report: StructuredReport
    sources: List[SourceDocument]
    summary: ResearchSummary
    trace: List[AgentTraceEvent]


# ---------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------

class ResearchError(Exception):
    """Base exception for research pipeline failures."""


class RetrievalError(ResearchError):
    """Raised when research retrieval fails."""


class LLMGenerationError(ResearchError):
    """Raised when LLM generation fails."""


# ---------------------------------------------------------------------
# Utility Helpers
# ---------------------------------------------------------------------

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def truncate(text: str, max_chars: int) -> str:
    text = text or ""
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + "..."


def extract_json_object(raw_text: str) -> Dict[str, Any]:
    """
    Extract the first JSON object from model output.
    This makes the app resilient even when a model adds extra text.
    """
    if not raw_text:
        raise ValueError("Empty model response.")

    raw_text = raw_text.strip()

    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", raw_text, flags=re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in model response.")

    return json.loads(match.group(0))


def simple_sentence_split(text: str) -> List[str]:
    text = " ".join((text or "").split())
    if not text:
        return []

    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if len(s.strip()) > 40]


def keyword_score(sentence: str, topic: str) -> int:
    topic_terms = {
        t.lower()
        for t in re.findall(r"[a-zA-Z0-9]+", topic)
        if len(t) > 3
    }
    sentence_terms = {
        t.lower()
        for t in re.findall(r"[a-zA-Z0-9]+", sentence)
        if len(t) > 3
    }
    return len(topic_terms.intersection(sentence_terms))


# ---------------------------------------------------------------------
# LLM Client
# ---------------------------------------------------------------------

class LLMClient:
    """
    Thin wrapper around the OpenAI Responses API.

    If OPENAI_API_KEY is not available, the agents fall back to deterministic
    extractive logic so the app remains demoable.
    """

    def __init__(self) -> None:
        self.model = os.getenv("OPENAI_MODEL", "gpt-5.5")
        self.api_key = os.getenv("OPENAI_API_KEY")

        self.enabled = bool(self.api_key and OpenAI)

        if self.enabled:
            self.client = OpenAI(api_key=self.api_key)
        else:
            self.client = None
            logger.warning(
                "OPENAI_API_KEY not found or openai package missing. "
                "Using deterministic fallback generation."
            )

    async def generate_text(self, system_prompt: str, user_prompt: str) -> str:
        if not self.enabled or self.client is None:
            raise LLMGenerationError("LLM client is not configured.")

        try:
            response = await asyncio.to_thread(
                self.client.responses.create,
                model=self.model,
                instructions=system_prompt,
                input=user_prompt,
            )

            text = getattr(response, "output_text", None)

            if text:
                return text

            # Defensive fallback for SDK response shape changes.
            output = getattr(response, "output", None)
            if output:
                chunks = []
                for item in output:
                    content = getattr(item, "content", None)
                    if not content:
                        continue
                    for part in content:
                        value = getattr(part, "text", None)
                        if value:
                            chunks.append(value)
                if chunks:
                    return "\n".join(chunks)

            raise LLMGenerationError("Could not extract text from LLM response.")

        except Exception as exc:
            logger.exception("LLM generation failed.")
            raise LLMGenerationError(str(exc)) from exc

    async def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> Dict[str, Any]:
        raw = await self.generate_text(system_prompt, user_prompt)
        try:
            return extract_json_object(raw)
        except Exception as exc:
            logger.exception("Failed to parse model JSON.")
            raise LLMGenerationError(f"Model did not return valid JSON: {exc}") from exc


# ---------------------------------------------------------------------
# Trace
# ---------------------------------------------------------------------

class AgentTrace:
    def __init__(self) -> None:
        self.events: List[AgentTraceEvent] = []

    def add(self, agent: str, status: str, message: str) -> None:
        self.events.append(
            AgentTraceEvent(
                agent=agent,
                status=status,
                message=message,
                timestamp=utc_now(),
            )
        )


# ---------------------------------------------------------------------
# Agent 1: Research Agent
# ---------------------------------------------------------------------

class ResearchAgent:
    name = "Research Agent"

    def __init__(self) -> None:
        self.tavily_api_key = os.getenv("TAVILY_API_KEY")
        self.tavily_enabled = bool(self.tavily_api_key and TavilyClient)

        if self.tavily_enabled:
            self.tavily_client = TavilyClient(api_key=self.tavily_api_key)
        else:
            self.tavily_client = None
            logger.warning(
                "TAVILY_API_KEY not found or tavily-python package missing. "
                "Using Wikipedia fallback retrieval."
            )

    async def run(
        self,
        topic: str,
        max_sources: int,
        depth: ResearchDepth,
        trace: AgentTrace,
    ) -> List[SourceDocument]:
        trace.add(self.name, "started", f"Retrieving sources for topic: {topic}")

        try:
            if self.tavily_enabled:
                docs = await self._search_tavily(topic, max_sources, depth)
            else:
                docs = await self._search_wikipedia(topic, max_sources)

            docs = self._dedupe_and_filter(docs, max_sources)

            if not docs:
                trace.add(self.name, "warning", "No useful sources were retrieved.")
                return []

            trace.add(
                self.name,
                "completed",
                f"Retrieved {len(docs)} source documents.",
            )
            return docs

        except Exception as exc:
            trace.add(self.name, "failed", str(exc))
            logger.exception("Research Agent failed.")
            raise RetrievalError(str(exc)) from exc

    async def _search_tavily(
        self,
        topic: str,
        max_sources: int,
        depth: ResearchDepth,
    ) -> List[SourceDocument]:
        assert self.tavily_client is not None

        search_depth = "advanced" if depth in {ResearchDepth.standard, ResearchDepth.deep} else "basic"

        response = await asyncio.to_thread(
            self.tavily_client.search,
            query=topic,
            search_depth=search_depth,
            max_results=max_sources,
            include_answer=False,
            include_raw_content=True,
        )

        results = response.get("results", []) if isinstance(response, dict) else []

        docs: List[SourceDocument] = []

        for idx, item in enumerate(results, start=1):
            title = item.get("title") or f"Source {idx}"
            url = item.get("url") or ""
            content = (
                item.get("raw_content")
                or item.get("content")
                or item.get("snippet")
                or ""
            )
            score = float(item.get("score") or 0.0)

            if len(content.strip()) < 80:
                continue

            docs.append(
                SourceDocument(
                    source_id=f"S{len(docs) + 1}",
                    title=title,
                    url=url,
                    content=truncate(content, 5000),
                    score=score,
                    provider="tavily",
                )
            )

        return docs

    async def _search_wikipedia(
        self,
        topic: str,
        max_sources: int,
    ) -> List[SourceDocument]:
        return await asyncio.to_thread(
            self._search_wikipedia_sync,
            topic,
            max_sources,
        )

    def _search_wikipedia_sync(
        self,
        topic: str,
        max_sources: int,
    ) -> List[SourceDocument]:
        encoded_topic = urllib.parse.urlencode(
            {
                "action": "query",
                "list": "search",
                "srsearch": topic,
                "srlimit": max_sources,
                "format": "json",
            }
        )

        search_url = f"https://en.wikipedia.org/w/api.php?{encoded_topic}"
        search_data = self._http_get_json(search_url)

        search_results = (
            search_data
            .get("query", {})
            .get("search", [])
        )

        docs: List[SourceDocument] = []

        for item in search_results:
            page_id = item.get("pageid")
            title = item.get("title", "Wikipedia Source")

            if not page_id:
                continue

            page_params = urllib.parse.urlencode(
                {
                    "action": "query",
                    "prop": "extracts|info",
                    "explaintext": "1",
                    "inprop": "url",
                    "pageids": page_id,
                    "format": "json",
                }
            )

            page_url = f"https://en.wikipedia.org/w/api.php?{page_params}"
            page_data = self._http_get_json(page_url)

            page = (
                page_data
                .get("query", {})
                .get("pages", {})
                .get(str(page_id), {})
            )

            content = page.get("extract") or ""
            full_url = page.get("fullurl") or f"https://en.wikipedia.org/?curid={page_id}"

            if len(content.strip()) < 80:
                continue

            docs.append(
                SourceDocument(
                    source_id=f"S{len(docs) + 1}",
                    title=title,
                    url=full_url,
                    content=truncate(content, 5000),
                    score=0.5,
                    provider="wikipedia",
                )
            )

        return docs

    def _http_get_json(self, url: str) -> Dict[str, Any]:
        request = urllib.request.Request(
            url,
            headers={
                "User-Agent": "MultiAgentResearchDemo/1.0"
            },
        )

        with urllib.request.urlopen(request, timeout=12) as response:
            raw = response.read().decode("utf-8")

        return json.loads(raw)

    def _dedupe_and_filter(
        self,
        docs: List[SourceDocument],
        max_sources: int,
    ) -> List[SourceDocument]:
        seen_urls = set()
        cleaned: List[SourceDocument] = []

        for doc in docs:
            if not doc.url or doc.url in seen_urls:
                continue

            if len(doc.content.strip()) < 80:
                continue

            seen_urls.add(doc.url)
            cleaned.append(doc)

            if len(cleaned) >= max_sources:
                break

        for idx, doc in enumerate(cleaned, start=1):
            doc.source_id = f"S{idx}"

        return cleaned


# ---------------------------------------------------------------------
# Agent 2: Summarizer Agent
# ---------------------------------------------------------------------

class SummarizerAgent:
    name = "Summarizer Agent"

    def __init__(self, llm_client: LLMClient) -> None:
        self.llm = llm_client

    async def run(
        self,
        topic: str,
        sources: List[SourceDocument],
        trace: AgentTrace,
    ) -> ResearchSummary:
        trace.add(self.name, "started", "Condensing retrieved source material.")

        if not sources:
            trace.add(self.name, "warning", "No sources available to summarize.")
            return ResearchSummary(
                short_summary="I don't have enough information from retrieved sources to summarize this topic.",
                key_findings=[],
                important_evidence=[],
                open_questions=[
                    "No source documents were available.",
                    "Try a more specific topic or configure a stronger retrieval provider.",
                ],
                source_coverage="Insufficient source coverage.",
            )

        if self.llm.enabled:
            try:
                result = await self._llm_summary(topic, sources)
                trace.add(self.name, "completed", "Generated structured summary using LLM.")
                return result
            except Exception as exc:
                trace.add(
                    self.name,
                    "warning",
                    f"LLM summarization failed. Falling back to extractive summary. Error: {exc}",
                )

        result = self._fallback_summary(topic, sources)
        trace.add(self.name, "completed", "Generated fallback extractive summary.")
        return result

    async def _llm_summary(
        self,
        topic: str,
        sources: List[SourceDocument],
    ) -> ResearchSummary:
        source_pack = self._format_sources_for_prompt(sources)

        system_prompt = """
You are the Summarizer Agent in a multi-agent research system.

Your job:
- Condense retrieved source material.
- Use only the provided sources.
- Do not invent facts.
- Preserve uncertainty.
- Return valid JSON only.

JSON schema:
{
  "short_summary": "string",
  "key_findings": ["string"],
  "important_evidence": ["string"],
  "open_questions": ["string"],
  "source_coverage": "string"
}
""".strip()

        user_prompt = f"""
Topic:
{topic}

Retrieved sources:
{source_pack}

Create a concise research summary grounded only in these sources.
Mention source IDs like [S1], [S2] when referring to evidence.
Return JSON only.
""".strip()

        data = await self.llm.generate_json(system_prompt, user_prompt)

        return ResearchSummary(
            short_summary=str(data.get("short_summary", "")).strip(),
            key_findings=[str(x).strip() for x in data.get("key_findings", [])][:8],
            important_evidence=[str(x).strip() for x in data.get("important_evidence", [])][:8],
            open_questions=[str(x).strip() for x in data.get("open_questions", [])][:5],
            source_coverage=str(data.get("source_coverage", "")).strip(),
        )

    def _fallback_summary(
        self,
        topic: str,
        sources: List[SourceDocument],
    ) -> ResearchSummary:
        scored_sentences: List[tuple[int, str, str]] = []

        for source in sources:
            for sentence in simple_sentence_split(source.content):
                score = keyword_score(sentence, topic)
                scored_sentences.append((score, source.source_id, sentence))

        scored_sentences.sort(key=lambda x: x[0], reverse=True)

        top = scored_sentences[:8]

        key_findings = [
            f"{sentence} [{source_id}]"
            for _, source_id, sentence in top[:5]
        ]

        important_evidence = [
            f"{sentence} [{source_id}]"
            for _, source_id, sentence in top[5:8]
        ]

        short_summary = (
            " ".join([finding for finding in key_findings[:2]])
            if key_findings
            else "The retrieved sources contain limited usable information."
        )

        return ResearchSummary(
            short_summary=truncate(short_summary, 700),
            key_findings=key_findings,
            important_evidence=important_evidence,
            open_questions=[
                "Are there more recent sources that should be included?",
                "Are there industry-specific examples or data points missing?",
                "Do the retrieved sources represent multiple viewpoints?",
            ],
            source_coverage=(
                f"Summary based on {len(sources)} retrieved source document(s). "
                "Fallback mode uses extractive sentence selection."
            ),
        )

    def _format_sources_for_prompt(self, sources: List[SourceDocument]) -> str:
        blocks = []

        for source in sources:
            blocks.append(
                f"""
[{source.source_id}]
Title: {source.title}
URL: {source.url}
Content:
{truncate(source.content, 3500)}
""".strip()
            )

        return "\n\n".join(blocks)


# ---------------------------------------------------------------------
# Agent 3: Report Generator Agent
# ---------------------------------------------------------------------

class ReportGeneratorAgent:
    name = "Report Generator Agent"

    def __init__(self, llm_client: LLMClient) -> None:
        self.llm = llm_client

    async def run(
        self,
        topic: str,
        sources: List[SourceDocument],
        summary: ResearchSummary,
        trace: AgentTrace,
    ) -> StructuredReport:
        trace.add(self.name, "started", "Generating structured report.")

        if not sources:
            trace.add(self.name, "warning", "No sources available for report generation.")
            return StructuredReport(
                executive_summary=(
                    "I don't have enough information from retrieved context to produce "
                    "a grounded research report."
                ),
                findings=[],
                risks=[
                    "No source documents were retrieved.",
                    "Any generated answer would risk being unsupported.",
                ],
                recommendations=[
                    "Configure Tavily search or provide a more specific topic.",
                    "Retry after confirming network access.",
                ],
                limitations=[
                    "The report is intentionally limited because retrieved context was insufficient."
                ],
            )

        if self.llm.enabled:
            try:
                result = await self._llm_report(topic, sources, summary)
                trace.add(self.name, "completed", "Generated structured report using LLM.")
                return result
            except Exception as exc:
                trace.add(
                    self.name,
                    "warning",
                    f"LLM report generation failed. Falling back to deterministic report. Error: {exc}",
                )

        result = self._fallback_report(topic, sources, summary)
        trace.add(self.name, "completed", "Generated fallback structured report.")
        return result

    async def _llm_report(
        self,
        topic: str,
        sources: List[SourceDocument],
        summary: ResearchSummary,
    ) -> StructuredReport:
        source_list = "\n".join(
            [
                f"[{s.source_id}] {s.title} - {s.url}"
                for s in sources
            ]
        )

        summary_json = summary.model_dump_json(indent=2)

        system_prompt = """
You are the Report Generator Agent in a multi-agent research system.

Your job:
- Produce a structured business-style report.
- Use only the provided summary and source list.
- Do not invent facts, names, statistics, dates, or claims.
- Cite evidence using source IDs like [S1].
- If evidence is weak, state the limitation clearly.
- Return valid JSON only.

JSON schema:
{
  "executive_summary": "string",
  "findings": ["string"],
  "risks": ["string"],
  "recommendations": ["string"],
  "limitations": ["string"]
}
""".strip()

        user_prompt = f"""
Research topic:
{topic}

Source list:
{source_list}

Summarizer Agent output:
{summary_json}

Generate a structured report with:
1. Executive summary
2. Findings
3. Risks
4. Recommendations
5. Limitations

Return JSON only.
""".strip()

        data = await self.llm.generate_json(system_prompt, user_prompt)

        return StructuredReport(
            executive_summary=str(data.get("executive_summary", "")).strip(),
            findings=[str(x).strip() for x in data.get("findings", [])][:10],
            risks=[str(x).strip() for x in data.get("risks", [])][:8],
            recommendations=[str(x).strip() for x in data.get("recommendations", [])][:8],
            limitations=[str(x).strip() for x in data.get("limitations", [])][:6],
        )

    def _fallback_report(
        self,
        topic: str,
        sources: List[SourceDocument],
        summary: ResearchSummary,
    ) -> StructuredReport:
        findings = summary.key_findings[:6]

        risks = [
            "The retrieved sources may not fully represent the latest developments.",
            "The fallback report generator cannot deeply reason across conflicting sources.",
            "Important domain-specific risks may be missing if they were not present in retrieved text.",
        ]

        recommendations = [
            "Review the cited source documents before making decisions.",
            "Run the research again with a narrower topic for higher precision.",
            "Add more retrieval providers for broader source coverage.",
            "Use the LLM mode for better synthesis and executive-level framing.",
        ]

        limitations = [
            summary.source_coverage,
            "Fallback mode uses extractive summarization, not deep reasoning.",
        ]

        executive_summary = (
            f"This report summarizes retrieved information about '{topic}'. "
            f"{summary.short_summary}"
        )

        return StructuredReport(
            executive_summary=truncate(executive_summary, 1200),
            findings=findings,
            risks=risks,
            recommendations=recommendations,
            limitations=limitations,
        )


# ---------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------

class MultiAgentResearchOrchestrator:
    """
    Coordinates the full agent workflow:

    1. Research Agent retrieves documents.
    2. Summarizer Agent condenses evidence.
    3. Report Generator Agent produces structured report.
    """

    def __init__(self) -> None:
        self.llm_client = LLMClient()
        self.research_agent = ResearchAgent()
        self.summarizer_agent = SummarizerAgent(self.llm_client)
        self.report_agent = ReportGeneratorAgent(self.llm_client)

    async def run(self, request: ResearchRequest) -> ResearchResponse:
        trace = AgentTrace()

        trace.add(
            "Orchestrator",
            "started",
            f"Starting multi-agent workflow for topic: {request.topic}",
        )

        try:
            sources = await self.research_agent.run(
                topic=request.topic,
                max_sources=request.max_sources,
                depth=request.depth,
                trace=trace,
            )

            summary = await self.summarizer_agent.run(
                topic=request.topic,
                sources=sources,
                trace=trace,
            )

            report = await self.report_agent.run(
                topic=request.topic,
                sources=sources,
                summary=summary,
                trace=trace,
            )

            trace.add("Orchestrator", "completed", "Workflow completed successfully.")

            return ResearchResponse(
                topic=request.topic,
                created_at=utc_now(),
                report=report,
                sources=sources,
                summary=summary,
                trace=trace.events,
            )

        except RetrievalError as exc:
            trace.add("Orchestrator", "failed", f"Retrieval failed: {exc}")
            raise

        except Exception as exc:
            trace.add("Orchestrator", "failed", f"Unexpected failure: {exc}")
            logger.exception("Orchestration failed.")
            raise


# ---------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------

app = FastAPI(
    title="Multi-Agent Research Report API",
    version="1.0.0",
    description=(
        "A modular multi-agent API that researches a topic, summarizes evidence, "
        "and generates a structured report."
    ),
)

orchestrator = MultiAgentResearchOrchestrator()


@app.get("/health")
async def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "timestamp": utc_now(),
        "openai_enabled": orchestrator.llm_client.enabled,
        "tavily_enabled": orchestrator.research_agent.tavily_enabled,
    }


@app.post("/research", response_model=ResearchResponse)
async def research(request: ResearchRequest) -> ResearchResponse:
    try:
        return await orchestrator.run(request)

    except RetrievalError as exc:
        raise HTTPException(
            status_code=502,
            detail={
                "error": "retrieval_failed",
                "message": str(exc),
            },
        ) from exc

    except ResearchError as exc:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "research_pipeline_failed",
                "message": str(exc),
            },
        ) from exc

    except Exception as exc:
        logger.exception("Unhandled API error.")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "internal_server_error",
                "message": "Unexpected server error.",
            },
        ) from exc


# ---------------------------------------------------------------------
# Local CLI Demo
# ---------------------------------------------------------------------

async def _demo() -> None:
    request = ResearchRequest(
        topic="How agentic AI will change cybersecurity operations",
        max_sources=5,
        depth=ResearchDepth.standard,
    )

    response = await orchestrator.run(request)
    print(response.model_dump_json(indent=2))


if __name__ == "__main__":
    asyncio.run(_demo())


# curl -X POST "http://127.0.0.1:8000/research" \
#   -H "Content-Type: application/json" \
#   -d '{
#     "topic": "How agentic AI will change cybersecurity operations",
#     "max_sources": 5,
#     "depth": "standard"
#   }'