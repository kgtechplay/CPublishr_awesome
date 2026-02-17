from __future__ import annotations

import json
import os
from typing import Any

import httpx

from src.contracts.prd import ResearchTrendResponse
from src.services.llm.azure_openai import AzureOpenAIClient, parse_json_object
from src.services.orchestration.contracts import NodeExecutionContext, NodeExecutionResult
from src.services.orchestration.nodes.base import OrchestrationNode


DEFAULT_RESEARCH_PROMPTS: dict[str, str] = {
    "entity_analysis_system": "You are an analyst extracting entities and context from user topic inputs.",
    "entity_analysis_user": (
        "Analyze the input and return strict JSON with keys: "
        "subjects (string[]), actions (string[]), context (string), related_entities (string[]).\n"
        "topic_title={topic_title}\ncore_idea={core_idea}\nuser_content={user_content}"
    ),
    "tools_system": "You identify competitors, alternatives, and adjacent entities for a topic.",
    "tools_user": (
        "Given main subjects and context, return strict JSON with keys: "
        "emerging_tools (string[]), competitor_entities (string[]), contrarian_angles (string[]).\n"
        "subjects={subjects}\nactions={actions}\ncontext={context}"
    ),
    "social_trends_system": "You summarize social media trend signals objectively.",
    "social_trends_user": (
        "Given search snippets, return strict JSON with keys: "
        "recent_discussions (string[]), key_insights (string[]).\n"
        "topic={topic}\nsearch_snippets={search_snippets}"
    ),
    "research_output_system": "You synthesize research into a strict output schema.",
    "research_output_user": (
        "Return strict JSON with keys: research_summary (string), emerging_tools (string[]), "
        "recent_discussions (string[]), key_insights (string[]), contrarian_angles (string[]).\n"
        "topic={topic}\nentity_analysis={entity_analysis}\nweb_search={web_search}\n"
        "tools={tools}\nsocial_trends={social_trends}"
    ),
}


class ResearchTrendsNode(OrchestrationNode):
    name = "research_trends"

    def __init__(self, llm: AzureOpenAIClient | None = None, prompts: dict[str, str] | None = None):
        self.llm = llm
        # Exposed prompt map so users can override/update prompts externally.
        self.prompts = prompts or DEFAULT_RESEARCH_PROMPTS.copy()

    def _fallback_output(self, topic: str) -> dict[str, Any]:
        return {
            "research_summary": f"Research summary for {topic}.",
            "emerging_tools": ["agents", "vector-db", "orchestrators"],
            "recent_discussions": ["quality vs speed", "human-in-loop"],
            "key_insights": ["master document reduces drift"],
            "contrarian_angles": ["more agents is not always better"],
        }

    def _llm_json(self, *, system_key: str, user_key: str, **kwargs: Any) -> dict[str, Any]:
        if not self.llm or not self.llm.enabled:
            return {}
        raw = self.llm.chat(
            system_prompt=self.prompts[system_key],
            user_prompt=self.prompts[user_key].format(**kwargs),
            temperature=0.2,
        )
        return parse_json_object(raw)

    def _tavily_search(self, *, query: str, max_results: int = 5) -> list[dict[str, Any]]:
        api_key = os.getenv("TAVILY_API_KEY", "").strip()
        if not api_key:
            return []

        payload = {
            "api_key": api_key,
            "query": query,
            "search_depth": "advanced",
            "max_results": max_results,
            "include_answer": False,
            "include_raw_content": False,
        }
        try:
            with httpx.Client(timeout=30.0) as client:
                resp = client.post("https://api.tavily.com/search", json=payload)
                resp.raise_for_status()
                data = resp.json()
            results = data.get("results") or []
            if not isinstance(results, list):
                return []
            return [r for r in results if isinstance(r, dict)]
        except Exception:
            return []

    # 1) entity_analysis
    def entity_analysis(self, *, topic_title: str, core_idea: str, user_content: str | None) -> dict[str, Any]:
        parsed = {}
        try:
            parsed = self._llm_json(
                system_key="entity_analysis_system",
                user_key="entity_analysis_user",
                topic_title=topic_title,
                core_idea=core_idea,
                user_content=user_content or "",
            )
        except Exception:
            parsed = {}

        subjects = [str(x) for x in (parsed.get("subjects") or []) if str(x).strip()]
        actions = [str(x) for x in (parsed.get("actions") or []) if str(x).strip()]
        related = [str(x) for x in (parsed.get("related_entities") or []) if str(x).strip()]
        context = str(parsed.get("context") or core_idea or topic_title).strip()

        if not subjects:
            subjects = [topic_title]
        if not actions:
            actions = ["analyze", "compare", "optimize"]

        return {
            "subjects": subjects,
            "actions": actions,
            "context": context,
            "related_entities": related,
        }

    # 2) web_search
    def web_search(self, *, entity_info: dict[str, Any], topic: str) -> dict[str, Any]:
        subjects = entity_info.get("subjects") or [topic]
        actions = entity_info.get("actions") or []
        queries = [
            f"{topic} latest trends",
            f"{' '.join(subjects[:3])} {' '.join(actions[:3])}".strip(),
        ]

        all_results: list[dict[str, Any]] = []
        for q in queries:
            all_results.extend(self._tavily_search(query=q, max_results=5))

        snippets = [str(r.get("content") or r.get("title") or "").strip() for r in all_results]
        snippets = [s for s in snippets if s]
        return {
            "queries": queries,
            "results": all_results,
            "snippets": snippets,
        }

    # 3) tools
    def tools(self, *, entity_info: dict[str, Any], web_data: dict[str, Any]) -> dict[str, Any]:
        parsed = {}
        try:
            parsed = self._llm_json(
                system_key="tools_system",
                user_key="tools_user",
                subjects=json.dumps(entity_info.get("subjects") or []),
                actions=json.dumps(entity_info.get("actions") or []),
                context=str(entity_info.get("context") or ""),
            )
        except Exception:
            parsed = {}

        emerging_tools = [str(x) for x in (parsed.get("emerging_tools") or []) if str(x).strip()]
        competitors = [str(x) for x in (parsed.get("competitor_entities") or []) if str(x).strip()]
        contrarian = [str(x) for x in (parsed.get("contrarian_angles") or []) if str(x).strip()]

        if not emerging_tools:
            emerging_tools = ["agents", "orchestrators", "evaluation frameworks"]
        if not contrarian:
            contrarian = ["More complexity does not always produce better outcomes."]

        return {
            "emerging_tools": emerging_tools,
            "competitor_entities": competitors,
            "contrarian_angles": contrarian,
            "supporting_results": web_data.get("results") or [],
        }

    # 4) Social_trends
    def social_trends(self, *, topic: str, web_data: dict[str, Any]) -> dict[str, Any]:
        parsed = {}
        try:
            parsed = self._llm_json(
                system_key="social_trends_system",
                user_key="social_trends_user",
                topic=topic,
                search_snippets=json.dumps((web_data.get("snippets") or [])[:15]),
            )
        except Exception:
            parsed = {}

        recent_discussions = [str(x) for x in (parsed.get("recent_discussions") or []) if str(x).strip()]
        key_insights = [str(x) for x in (parsed.get("key_insights") or []) if str(x).strip()]

        if not recent_discussions:
            recent_discussions = [
                "Reliability vs iteration speed",
                "Human-in-the-loop checkpoints",
            ]
        if not key_insights:
            key_insights = ["Teams with structured content pipelines report lower message drift."]

        return {
            "recent_discussions": recent_discussions,
            "key_insights": key_insights,
        }

    # 5) Research_output
    def research_output(
        self,
        *,
        topic: str,
        entity_data: dict[str, Any],
        web_data: dict[str, Any],
        tools_data: dict[str, Any],
        social_data: dict[str, Any],
    ) -> dict[str, Any]:
        fallback = self._fallback_output(topic)

        parsed = {}
        try:
            parsed = self._llm_json(
                system_key="research_output_system",
                user_key="research_output_user",
                topic=topic,
                entity_analysis=json.dumps(entity_data),
                web_search=json.dumps({"queries": web_data.get("queries"), "snippets": web_data.get("snippets", [])[:15]}),
                tools=json.dumps({
                    "emerging_tools": tools_data.get("emerging_tools"),
                    "competitor_entities": tools_data.get("competitor_entities"),
                    "contrarian_angles": tools_data.get("contrarian_angles"),
                }),
                social_trends=json.dumps(social_data),
            )
        except Exception:
            parsed = {}

        output = {
            "research_summary": str(parsed.get("research_summary") or fallback["research_summary"]),
            "emerging_tools": [str(x) for x in (parsed.get("emerging_tools") or tools_data.get("emerging_tools") or fallback["emerging_tools"])],
            "recent_discussions": [str(x) for x in (parsed.get("recent_discussions") or social_data.get("recent_discussions") or fallback["recent_discussions"])],
            "key_insights": [str(x) for x in (parsed.get("key_insights") or social_data.get("key_insights") or fallback["key_insights"])],
            "contrarian_angles": [str(x) for x in (parsed.get("contrarian_angles") or tools_data.get("contrarian_angles") or fallback["contrarian_angles"])],
        }
        return ResearchTrendResponse.model_validate(output).model_dump()

    def run(self, context: NodeExecutionContext) -> NodeExecutionResult:
        b = context.state.get("context_bundle") or {}
        topic_title = str(b.get("topic_title") or "AI Topic").strip()
        topic = str(b.get("normalized_topic") or topic_title or "ai").strip()
        core_idea = str(b.get("core_idea") or "").strip()
        user_content = b.get("user_content")

        fallback = self._fallback_output(topic)

        try:
            entity_data = self.entity_analysis(topic_title=topic_title, core_idea=core_idea, user_content=user_content)
            web_data = self.web_search(entity_info=entity_data, topic=topic)
            tools_data = self.tools(entity_info=entity_data, web_data=web_data)
            social_data = self.social_trends(topic=topic, web_data=web_data)
            output = self.research_output(
                topic=topic,
                entity_data=entity_data,
                web_data=web_data,
                tools_data=tools_data,
                social_data=social_data,
            )
        except Exception:
            output = ResearchTrendResponse.model_validate(fallback).model_dump()

        context.state["research"] = output
        return NodeExecutionResult(status="completed", output_payload=output)
