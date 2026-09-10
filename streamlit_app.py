"""Minimal Streamlit code review assistant - no custom theme, few dependencies.

Run locally:        streamlit run streamlit_app.py
Streamlit Cloud:    main file = example/codereview_simple/streamlit_app.py
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from review_engine import (
    ARCHITECTURE_EDGES,
    ARCHITECTURE_NODES,
    DEFAULT_LANGSMITH_PROJECT,
    DEFAULT_MODEL,
    LANGSMITH_ENDPOINT,
    LANGUAGES,
    MODEL_CHOICES,
    SAMPLE_SNIPPETS,
    UPLOAD_EXTENSIONS,
    LangSmithConfig,
    ReviewError,
    analyze,
    best_practices,
    category_counts,
    code_metrics,
    health_score,
    language_for_path,
    load_usage,
    offline_review,
    review_code,
    severity_counts,
    unit_label,
    usage_summary,
)

LANGCHAIN_WIRING = """from langchain_classic.agents import (
    AgentExecutor, create_tool_calling_agent)
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

tools = [static_code_check]
agent = create_tool_calling_agent(llm, tools, prompt)
executor = AgentExecutor(
    agent=agent, tools=tools, verbose=True,
    return_intermediate_steps=True, max_iterations=4,
)
result = executor.invoke({"input": ...})"""

st.set_page_config(page_title="Code Review Assistant", page_icon="🔍", layout="wide")


def findings_frame(findings) -> pd.DataFrame:
    return pd.DataFrame([{
        "Severity": f.severity.upper(), "Line": f.line, "Category": f.category,
        "Finding": f.message, "Recommendation": f.recommendation,
    } for f in findings])


st.title("Code Review Assistant")
st.caption("Deterministic static analysis combined with an LLM review, "
           "orchestrated by a LangChain tool-calling agent.")

# ------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Configuration")
    language_label = st.selectbox("Language", list(LANGUAGES), index=0)
    language = LANGUAGES[language_label]
    model = st.selectbox("Model", MODEL_CHOICES, index=MODEL_CHOICES.index(DEFAULT_MODEL))
    api_key = st.text_input("OpenAI API key", type="password",
                            help="Kept in this session only; never written to disk.")
    uploaded = st.file_uploader("Upload source file",
                                type=[ext.lstrip(".") for ext in UPLOAD_EXTENSIONS])

    st.subheader("LangSmith")
    trace_enabled = st.checkbox("Enable tracing & evaluation", value=False)
    langsmith_key = st.text_input("LangSmith API key", type="password")
    langsmith_project = st.text_input("Project", value=DEFAULT_LANGSMITH_PROJECT)
    langsmith_endpoint = st.text_input("Endpoint", value=LANGSMITH_ENDPOINT)
    st.caption("Static analysis runs locally and needs no keys.")

langsmith = LangSmithConfig(enabled=trace_enabled, api_key=langsmith_key,
                            project=langsmith_project, endpoint=langsmith_endpoint)


def key_errors() -> list[str]:
    """Blocking validation shown before any agent action runs."""
    problems = []
    if not api_key.strip():
        problems.append("OpenAI API key is missing - add it in the sidebar to run an AI review.")
    if trace_enabled and not langsmith_key.strip():
        problems.append("LangSmith API key is missing while tracing is enabled - "
                        "add the key or turn tracing off.")
    return problems


# -------------------------------------------------------------------- editor
if uploaded is not None:
    with st.spinner(f"Loading {uploaded.name}..."):
        code = uploaded.getvalue().decode("utf-8", errors="replace")
        language = language_for_path(uploaded.name)
    st.sidebar.success(f"Loaded {uploaded.name} ({language})")
else:
    code = st.session_state.get("code", SAMPLE_SNIPPETS[language])
    if st.session_state.get("lang") != language:
        code = SAMPLE_SNIPPETS[language]
    st.session_state["lang"] = language

editor, workspace = st.columns([2, 5], gap="small")

with editor:
    st.subheader("Source under review")
    code = st.text_area("Code", value=code, height=520, label_visibility="collapsed")
    st.session_state["code"] = code
    run_review = st.button("Run AI Review", type="primary", use_container_width=True)
    for problem in key_errors():
        st.error(problem)

findings = analyze(code, language) if code.strip() else []
counts = severity_counts(findings)

with workspace:
    tab_analysis, tab_ai, tab_arch, tab_monitor = st.tabs(
        ["Code review analysis", "AI review", "Agentic architecture",
         "Monitoring & evaluation"])

    # ---------------------------------------------------- tab 1: analysis
    with tab_analysis:
        metrics = code_metrics(code, language)
        summary = st.columns(4)
        summary[0].metric("Findings", len(findings))
        summary[1].metric("Health score", f"{health_score(findings)}/100")
        summary[2].metric(unit_label(language), int(metrics["units"]))
        summary[3].metric("Documented", f"{metrics['doc_coverage']:.0f}%")

        left, right = st.columns(2)
        with left:
            st.markdown("**Findings by severity**")
            st.bar_chart(pd.DataFrame({"Findings": list(counts.values())},
                                      index=[s.capitalize() for s in counts]))
        with right:
            st.markdown("**Findings by category**")
            categories = category_counts(findings)
            if categories:
                st.bar_chart(pd.DataFrame({"Findings": list(categories.values())},
                                          index=list(categories)),
                             horizontal=True)
            else:
                st.success("No issues detected by the static checks.")

        st.markdown("**Findings**")
        chosen = st.multiselect("Severity filter", ["high", "medium", "low"],
                                default=["high", "medium", "low"],
                                label_visibility="collapsed")
        visible = [f for f in findings if f.severity in chosen]
        if visible:
            st.dataframe(findings_frame(visible), hide_index=True, height=250)
        else:
            st.info("No findings for the selected severities.")

        detail_left, detail_right = st.columns(2)
        with detail_left:
            st.markdown("**Recommended actions**")
            actions = sorted({f.recommendation for f in findings if f.recommendation})
            st.markdown("\n".join(f"- {a}" for a in actions) or "- Nothing to action.")
        with detail_right:
            st.markdown(f"**{language.title()} best practices**")
            st.markdown("\n".join(f"- {p}" for p in best_practices(language)))

    # ----------------------------------------------------- tab 2: AI review
    with tab_ai:
        if run_review:
            problems = key_errors()
            if problems:
                for problem in problems:
                    st.error(problem)
            elif not code.strip():
                st.error("Add a snippet before requesting a review.")
            else:
                with st.spinner("Agent running - static check, then LLM review..."):
                    try:
                        st.session_state["review"] = review_code(
                            code, language, api_key=api_key, model=model,
                            langsmith=langsmith)
                    except ReviewError as exc:
                        st.session_state["review"] = None
                        st.error(str(exc))
                        st.code(offline_review(code, language))

        result = st.session_state.get("review")
        if result:
            meta = st.columns(5)
            meta[0].metric("Model", result.model)
            meta[1].metric("Latency", f"{result.duration_s}s")
            meta[2].metric("Tokens", result.total_tokens)
            meta[3].metric("LLM calls", result.llm_calls)
            meta[4].metric("Tool calls", result.tool_calls)
            if result.trace_url:
                st.markdown(f"[Open this run in LangSmith]({result.trace_url})")
            st.markdown(result.text)
        elif not run_review:
            st.info("Press **Run AI Review** to turn the static findings into a "
                    "Summary / Must fix / Suggestions review.")

    # ------------------------------------------- tab 3: agentic architecture
    with tab_arch:
        st.markdown("**AgentExecutor loop over a deterministic checker**")
        lines = ["digraph {", "  rankdir=TB; node [shape=box style=rounded];"]
        for key, title, subtitle in ARCHITECTURE_NODES:
            lines.append(f'  {key} [label="{title}\\n{subtitle}"];')
        for source, target, label in ARCHITECTURE_EDGES:
            lines.append(f'  {source} -> {target} [label="{label}" fontsize=8];')
        lines.append("}")

        diagram, wiring = st.columns([3, 2])
        with diagram:
            st.graphviz_chart("\n".join(lines))
        with wiring:
            st.markdown("**LangChain wiring**")
            st.code(LANGCHAIN_WIRING, language="python")

        st.markdown("**Agent trace (last run)**")
        if result and result.steps:
            for index, step in enumerate(result.steps, 1):
                with st.expander(f"{index}. {step.kind} - {step.name}", expanded=index == 1):
                    st.code(step.detail)
        else:
            st.info("Run an AI review to capture the executor's intermediate steps.")

        st.markdown("**Component responsibilities**")
        st.dataframe(pd.DataFrame(ARCHITECTURE_NODES,
                                  columns=["Id", "Component", "Responsibility"]),
                     hide_index=True)

    # -------------------------------------------- tab 4: monitoring & eval
    with tab_monitor:
        records = load_usage()
        stats = usage_summary(records)
        kpis = st.columns(6)
        kpis[0].metric("Total runs", stats["runs"])
        kpis[1].metric("Success rate", f"{stats['success_rate']:.0f}%")
        kpis[2].metric("Avg latency", f"{stats['avg_latency']:.1f}s")
        kpis[3].metric("Tokens used", int(stats["total_tokens"]))
        kpis[4].metric("Avg eval score", f"{stats['avg_eval']:.2f}")
        kpis[5].metric("Traced runs", int(stats["traced"]))

        st.markdown("**LangSmith evaluation**")
        if langsmith.active:
            st.success(f"Tracing to project '{langsmith.project}' at {langsmith.endpoint}")
        elif langsmith.enabled:
            st.warning("Enter a LangSmith API key in the sidebar to publish traces.")
        else:
            st.info("Enable LangSmith tracing in the sidebar to publish runs and feedback.")

        if result and result.scores:
            st.bar_chart(pd.DataFrame({"Score": list(result.scores.values())},
                                      index=list(result.scores)))
            st.caption("Evaluators: section structure, finding coverage, grounding in "
                       "categories, actionability, conciseness. Scores are pushed to "
                       "LangSmith as run feedback when tracing is enabled.")

        if records:
            usage_df = pd.DataFrame(records)
            usage_df["timestamp"] = pd.to_datetime(usage_df["timestamp"])
            by_language = usage_df.groupby("language").size()
            st.markdown("**Runs by language**")
            st.bar_chart(by_language.rename("Runs"))
            st.markdown("**Run history**")
            st.dataframe(usage_df.sort_values("timestamp", ascending=False),
                         hide_index=True, height=260)
        else:
            st.info("No runs recorded yet. Every AI review appends a row to usage_log.jsonl.")
