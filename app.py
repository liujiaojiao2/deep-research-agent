"""DeepResearch Agent — Streamlit 互动 Demo。"""
from __future__ import annotations

import os
import sys
import time
from datetime import datetime

import streamlit as st

sys.path.insert(0, os.path.dirname(__file__))

from src.graph import build_main_graph

# ---------- page config ----------
st.set_page_config(
    page_title="DeepResearch Agent Demo",
    page_icon="🔬",
    layout="wide",
)

st.title("🔬 DeepResearch Agent")
st.caption("Multi-Agent 深度研究报告生成系统 · LangGraph · Supervisor 路由 · 自进化闭环")

# ---------- sidebar: 配置 ----------
with st.sidebar:
    st.header("⚙️ 配置")
    mode = st.selectbox(
        "Researcher 模式",
        options=["react", "rewoo"],
        index=0,
        help="ReAct: LLM 自主选工具（N+1 次 LLM 调用）；ReWOO: 一次规划 + 纯执行（1 次 LLM 调用）",
    )
    quality_threshold = st.slider(
        "质量阈值",
        min_value=5.0,
        max_value=9.5,
        value=7.0,
        step=0.5,
        help="quality_eval 评分超过此值则直接进入终稿，否则触发 red_team → revision 自进化",
    )
    max_iterations = st.slider(
        "最大迭代次数",
        min_value=1,
        max_value=5,
        value=3,
        help="自进化（red_team → revision → re-eval）的最大轮次",
    )
    st.divider()
    st.caption("环境变量配置在运行前注入，不影响全局设置。")

# ---------- query input ----------
query = st.text_area(
    "🔍 研究问题",
    value="Transformer 架构中多头注意力机制的原理是什么？",
    height=80,
    placeholder="输入你想研究的问题...",
)

# ---------- run ----------
if st.button("🚀 开始研究", type="primary", use_container_width=True):
    if not query.strip():
        st.warning("请输入研究问题")
        st.stop()

    # configure env
    os.environ["RESEARCHER_MODE"] = mode
    os.environ["QUALITY_THRESHOLD"] = str(quality_threshold)

    # build graph
    graph = build_main_graph(interactive=False)

    initial_state = {
        "query": query.strip(),
        "research_brief": "",
        "research_results": [],
        "draft_report": "",
        "final_report": "",
        "quality_score": {},
        "red_team_feedback": "",
        "iteration_count": 0,
        "max_iterations": max_iterations,
        "next_agent": "",
        "is_complete": False,
        "messages": [],
    }

    # ---------- progress containers ----------
    progress_bar = st.progress(0, text="初始化...")
    status_area = st.empty()

    # containers for each phase
    brief_col = st.empty()
    research_col = st.empty()
    draft_col = st.empty()
    quality_col = st.empty()
    redteam_col = st.empty()
    revision_col = st.empty()
    final_col = st.empty()

    # step tracker
    steps: list[dict] = []
    phase_labels = {
        "supervisor": "🧭 Supervisor 决策路由",
        "brief_writer": "📋 生成研究简报",
        "researcher": "🔍 多源信息检索",
        "draft_writer": "✍️ 撰写初稿",
        "quality_eval": "📊 质量评估",
        "red_team": "🛡️ Red Team 对抗审查",
        "revision": "🔧 补充搜索 + 修订报告",
        "final_report": "📄 生成终稿",
        "memory_archive": "💾 记忆归档",
        "evolution_log": "🧬 HarnessForge 进化记录",
        "skill_library": "📚 Memento-Skills 技能提取",
    }

    start_time = time.monotonic()
    final_state = initial_state

    try:
        for event in graph.stream(initial_state, config={"recursion_limit": 50}):
            for node_name, update in event.items():
                final_state = {**final_state, **(update or {})}

                if node_name in phase_labels:
                    ts = datetime.now().strftime("%H:%M:%S")
                    summary = ""
                    if node_name == "supervisor":
                        next_a = update.get("next_agent", "") if update else ""
                        summary = f"→ 路由到 **{next_a}**"
                    elif node_name == "brief_writer":
                        brief = final_state.get("research_brief", "")
                        summary = f"简报长度: {len(brief)} 字符"
                    elif node_name == "researcher":
                        results = final_state.get("research_results", [])
                        n = len(results)
                        sources = []
                        for r in results:
                            src = r.get("source", "")
                            if src and src not in sources:
                                sources.append(src)
                        src_list = ", ".join(sources[:4]) if sources else "—"
                        summary = f"检索到 {n} 条结果 | 来源: {src_list}"
                    elif node_name == "draft_writer":
                        draft = final_state.get("draft_report", "")
                        summary = f"初稿长度: {len(draft)} 字符"
                    elif node_name == "quality_eval":
                        qs = final_state.get("quality_score") or {}
                        overall = qs.get("overall", "?")
                        summary = f"总分: **{overall}** | 准确性: {qs.get('accuracy', '?')} | 完整性: {qs.get('completeness', '?')} | 引用: {qs.get('citation', '?')}"
                    elif node_name == "red_team":
                        feedback = final_state.get("red_team_feedback", "")[:120]
                        summary = f"发现缺陷: {feedback}..."
                    elif node_name == "revision":
                        draft = final_state.get("draft_report", "")
                        iterations = final_state.get("iteration_count", 0)
                        summary = f"修订后初稿长度: {len(draft)} 字符 | 当前迭代: {iterations}/{max_iterations}"
                    elif node_name == "final_report":
                        final_text = final_state.get("final_report", "")
                        summary = f"终稿长度: {len(final_text)} 字符"

                    steps.append({
                        "ts": ts,
                        "node": node_name,
                        "label": phase_labels.get(node_name, node_name),
                        "summary": summary,
                    })

                    # update progress
                    progress = min(len(steps) / 12, 1.0)
                    progress_bar.progress(progress, text=f"Step {len(steps)}: {phase_labels.get(node_name, node_name)}")

                    # show current step highlights
                    with status_area.container():
                        for step in steps[-5:]:
                            icon = "✅" if step["node"] != "supervisor" else "🧭"
                            st.markdown(f"{icon} `{step['ts']}` **{step['label']}** — {step['summary']}")

        # ---------- 展开各阶段详细内容 ----------
        brief_text = final_state.get("research_brief") or ""
        if brief_text:
            with st.expander("📋 研究简报", expanded=False):
                st.markdown(brief_text)

        results = final_state.get("research_results", [])
        if results:
            with st.expander(f"🔍 检索结果 ({len(results)} 条)", expanded=False):
                for i, r in enumerate(results):
                    st.markdown(f"**{i+1}. {r.get('query', '?')}**")
                    st.caption(f"来源: {r.get('source', '?')}")
                    st.markdown(r.get("content", "")[:500] + ("..." if len(r.get("content", "")) > 500 else ""))
                    if i < len(results) - 1:
                        st.divider()

        quality_full = final_state.get("quality_score") or {}
        if quality_full:
            with st.expander("📊 质量评估详情", expanded=False):
                cols = st.columns(5)
                cols[0].metric("准确性", quality_full.get("accuracy", "?"))
                cols[1].metric("完整性", quality_full.get("completeness", "?"))
                cols[2].metric("逻辑性", quality_full.get("logic", "?"))
                cols[3].metric("引用", quality_full.get("citation", "?"))
                cols[4].metric("综合分", quality_full.get("overall", "?"))
                feedback = quality_full.get("feedback", "")
                if feedback:
                    st.caption(f"反馈: {feedback}")

        rt_feedback = final_state.get("red_team_feedback") or ""
        if rt_feedback:
            with st.expander("🛡️ Red Team 审查反馈", expanded=False):
                st.markdown(rt_feedback)

        # ---------- 最终报告 ----------
        elapsed = time.monotonic() - start_time
        st.divider()
        st.header("📄 最终研究报告")

        meta_cols = st.columns(4)
        meta_cols[0].metric("迭代次数", final_state.get("iteration_count", 0))
        meta_cols[1].metric("质量分", f'{quality_full.get("overall", "?"):.1f}' if isinstance(quality_full.get("overall"), (int, float)) else str(quality_full.get("overall", "?")))
        meta_cols[2].metric("用时", f"{elapsed:.0f}s")
        meta_cols[3].metric("Researcher", mode.upper())

        final_text = final_state.get("final_report") or final_state.get("draft_report") or "*报告生成失败*"
        st.markdown(final_text)

        # download button
        st.download_button(
            label="📥 下载 Markdown 报告",
            data=final_text,
            file_name=f"research_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown",
        )

    except Exception as e:
        st.error(f"运行出错: {type(e).__name__}: {e}")
        st.code(str(e))

else:
    # ---------- placeholder ----------
    st.info("👆 输入研究问题，点击「开始研究」查看多 Agent 协作的完整流程。")
    st.markdown("""
    ### 系统架构

    | 阶段 | 节点 | 职责 |
    |------|------|------|
    | 1 | 🧭 Supervisor | 纯函数状态机，根据当前进度决定下一步 |
    | 2 | 📋 Brief Writer | 子问题拆分 + 关键词提取 + 研究结构规划 |
    | 3 | 🔍 Researcher | 多源搜索（本地知识库 / Wikipedia / ArXiv / Web） |
    | 4 | ✍️ Draft Writer | 基于研究资料撰写结构化报告 |
    | 5 | 📊 Quality Eval | 5 维评分（准确性/完整性/逻辑/引用/综合） |
    | 6 | 🛡️ Red Team | 对抗审查发现幻觉和逻辑漏洞 |
    | 7 | 🔧 Revision | 补充搜索 + 重写报告修复发现的问题 |
    | 8 | 📄 Final Report | 润色生成终稿 |

    **质量闭环：** 低分自动进入 red_team → revision → re-eval 循环，直到达标或达到最大迭代次数。
    """)
