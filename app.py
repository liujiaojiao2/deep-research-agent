"""DeepResearch Agent — Streamlit 互动 Demo，含 Token 追踪与成本可视化。"""
from __future__ import annotations

import os
import sys
import time
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()

import streamlit as st

sys.path.insert(0, os.path.dirname(__file__))

from src.token_tracker import TokenTracker, get_tracker, reset_tracker, MODEL_PRICING
from src.obsidian import export_to_obsidian, is_obsidian_configured

# ---------- page config ----------
st.set_page_config(
    page_title="DeepResearch Agent Demo",
    page_icon="🔬",
    layout="wide",
)

st.title("🔬 DeepResearch Agent")
st.caption("Multi-Agent 深度研究报告生成系统 · LangGraph · Supervisor 路由 · 自进化闭环")


# ---------- inject token tracking into LLM pipeline ----------
def _install_token_tracker(tracker: TokenTracker):
    """Monkey-patch ChatOpenAI.invoke 类方法，所有 LLM 调用自动追踪。

    类级别 patch 自动覆盖 bind_tools / ReAct 内部调用等所有路径，
    无需逐个 agent 模块 patch。
    """
    tracker.install()


# ---------- sidebar: 配置 ----------
with st.sidebar:
    st.header("⚙️ 配置")
    mode = st.selectbox(
        "Researcher 模式",
        options=["react", "rewoo"],
        index=0,
        help="ReAct: LLM 自主选工具（N+1 次 LLM 调用）；ReWOO: 一次规划 + 纯执行（1 次 LLM 调用）",
    )
    search_provider = st.selectbox(
        "搜索引擎",
        options=["duckduckgo", "google"],
        index=0,
        help="DuckDuckGo 零配置免代理；Google 需网络能访问 google.com（可设 GOOGLE_PROXY 环境变量）",
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
    enrich_mindmap = st.checkbox(
        "🌳 思维导图 LLM 增强",
        value=False,
        help="给大纲每个 heading 补 3-5 个关键要点叶子（额外 LLM 调用，~2-4k output tokens）",
    )
    st.divider()
    obs_ok = is_obsidian_configured()
    if obs_ok:
        st.caption("📝 Obsidian 导出已配置")
    else:
        st.caption("📝 Obsidian 未配置（设 OBSIDIAN_VAULT_PATH 启用）")
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
    os.environ["SEARCH_PROVIDER"] = search_provider

    # init token tracker and install
    reset_tracker()
    tracker = get_tracker()
    _install_token_tracker(tracker)

    from src.graph import build_main_graph

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
    token_area = st.sidebar.empty()

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

                    progress = min(len(steps) / 12, 1.0)
                    progress_bar.progress(progress, text=f"Step {len(steps)}: {phase_labels.get(node_name, node_name)}")

                    with status_area.container():
                        for step in steps[-5:]:
                            icon = "✅" if step["node"] != "supervisor" else "🧭"
                            st.markdown(f"{icon} `{step['ts']}` **{step['label']}** — {step['summary']}")

                    # 实时更新 token 统计
                    with token_area.container():
                        total = tracker.total
                        st.metric("累计 Token", f"{total.total_tokens:,}")
                        st.metric("LLM 调用次数", len(tracker.records))
                        st.metric("预估成本", f"¥{tracker.total_cost_yuan:.4f}")

        # ---------- Token 使用详细面板 ----------
        with st.expander("💰 Token 用量与成本明细", expanded=True):
            c1, c2, c3 = st.columns(3)
            c1.metric("LLM 调用次数", len(tracker.records))
            c2.metric("输入 Token", f"{tracker.total.input_tokens:,}")
            c3.metric("输出 Token", f"{tracker.total.output_tokens:,}")

            c4, c5, c6 = st.columns(3)
            c4.metric("合计 Token", f"{tracker.total.total_tokens:,}")
            c5.metric("预估成本", f"¥{tracker.total_cost_yuan:.4f}")
            # 找到实际使用的 model name
            used_model = tracker.records[0].model if tracker.records else "unknown"
            c6.metric("模型", used_model)

            st.caption(
                f"DeepSeek 定价参考: {used_model} "
                f"输入 ¥{MODEL_PRICING.get(used_model, (1.0, 2.0))[0]}/1M tokens, "
                f"输出 ¥{MODEL_PRICING.get(used_model, (1.0, 2.0))[1]}/1M tokens"
            )

            if tracker.records:
                st.divider()
                st.caption("调用明细 (最近 20 条)")
                import pandas as pd
                rows = []
                for r in tracker.records[-20:]:
                    rows.append({
                        "模型": r.model,
                        "输入": f"{r.input_tokens:,}",
                        "输出": f"{r.output_tokens:,}",
                        "提示词预览": r.prompt_preview[:100],
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
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

        meta_cols = st.columns(5)
        meta_cols[0].metric("迭代次数", final_state.get("iteration_count", 0))
        meta_cols[1].metric("质量分",
            f'{quality_full.get("overall", "?"):.1f}'
            if isinstance(quality_full.get("overall"), (int, float))
            else str(quality_full.get("overall", "?")))
        meta_cols[2].metric("用时", f"{elapsed:.0f}s")
        meta_cols[3].metric("Researcher", mode.upper())
        meta_cols[4].metric("Token 成本", f"¥{tracker.total_cost_yuan:.4f}")

        final_text = final_state.get("final_report") or final_state.get("draft_report") or "*报告生成失败*"

        from src.tools.mindmap_tool import report_to_mindmap_html, report_to_outline
        import streamlit.components.v1 as components

        outline_md = report_to_outline(final_text, enrich=enrich_mindmap)
        mindmap_html = report_to_mindmap_html(
            final_text, enrich=enrich_mindmap, title=query.strip()[:60] or "MindMap"
        )

        tab_md, tab_outline, tab_mm = st.tabs(["📄 报告", "📝 大纲", "🌳 思维导图"])
        with tab_md:
            st.markdown(final_text)
        with tab_outline:
            st.code(outline_md, language="markdown")
        with tab_mm:
            components.html(mindmap_html, height=650, scrolling=True)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        dl_col1, dl_col2, dl_col3, obs_col = st.columns([1, 1, 1, 1])
        with dl_col1:
            st.download_button(
                label="📥 下载报告",
                data=final_text,
                file_name=f"research_{ts}.md",
                mime="text/markdown",
                use_container_width=True,
            )
        with dl_col2:
            st.download_button(
                label="📥 下载大纲",
                data=outline_md,
                file_name=f"outline_{ts}.md",
                mime="text/markdown",
                use_container_width=True,
            )
        with dl_col3:
            st.download_button(
                label="📥 下载思维导图",
                data=mindmap_html,
                file_name=f"mindmap_{ts}.html",
                mime="text/html",
                use_container_width=True,
            )
        with obs_col:
            if is_obsidian_configured():
                if st.button("📝 一键导出到 Obsidian", use_container_width=True):
                    path = export_to_obsidian(
                        content=final_text,
                        query=query.strip(),
                        quality_score=quality_full if quality_full else None,
                    )
                    if path:
                        st.success(f"已导出到 Obsidian: `{path}`")
                    else:
                        st.error("导出失败，请检查 Obsidian Vault 路径配置")
            else:
                st.button(
                    "📝 Obsidian 未配置",
                    disabled=True,
                    use_container_width=True,
                    help="请在 .env 中设置 OBSIDIAN_VAULT_PATH",
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
    **成本追踪：** 每次 LLM 调用自动统计 token 消耗，按 agent 拆分，实时计算 ¥ 成本。
    """)
