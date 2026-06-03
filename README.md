# DeepResearch Agent

基于 LangGraph 的多 Agent 深度研究系统，带自进化（Red Team / Quality / Revision）闭环。

- 主 LLM：**DeepSeek**（兼容 OpenAI 协议）
- 搜索：**DuckDuckGo**（零配置）
- 框架：**LangGraph 1.x**
- 运行环境：MacBook M3 Air，Python 3.11

---

## 架构

```
user query
   │
   ▼
┌─────────────┐
│ supervisor  │◀─────────────────────────┐
└──────┬──────┘                          │
       │ 条件路由                         │
       ├──▶ brief_writer ─────────────────┤
       ├──▶ researcher (DDG + compress) ──┤
       ├──▶ draft_writer ─────────────────┤
       ├──▶ quality_eval (多维 JSON 评分)─┤
       ├──▶ red_team → revision ──────────┘
       └──▶ final_report ─▶ END
```

Supervisor 决策表（按优先级短路）：

| 状态 | 路由 |
|---|---|
| 无 `research_brief` | `brief_writer` |
| 无 `research_results` | `researcher` |
| 无 `draft_report` | `draft_writer` |
| 无 `quality_score` | `quality_eval` |
| `overall >= 7.0` | `final_report` |
| `iteration_count >= max_iter` | `final_report`（兜底） |
| 其他 | `red_team` |

---

## 快速开始

```bash
# 1. 安装 uv（如已装可跳）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. 依赖
uv sync

# 3. 配置 .env（必填 DeepSeek key）
cp .env.example .env
# 编辑 .env，把 DEEPSEEK_API_KEY 改成你的真实 key

# 4. 环境黑箱探测
uv run python scripts/verify_env.py

# 5. 跑一份研究
uv run python main.py "你的研究问题" --max-iter 2
```

输出报告默认保存到 `outputs/report_YYYYMMDD_HHMMSS.md`。

---

## 调参

| 调什么 | 在哪改 | 默认 |
|---|---|---|
| 质量阈值（决定是否结稿） | `src/agents/supervisor_agent.py` `QUALITY_THRESHOLD` | `7.0` |
| 自进化迭代上限 | CLI `--max-iter` 或 `state.max_iterations` | `3` |
| LangGraph 递归上限 | CLI `--recursion-limit` | `50` |
| Researcher 单查询数 | `researcher_node(max_results_per_query=)` | `3` |
| 补搜索查询上限 | `src/agents/revision_agent.py` `MAX_SUPPLEMENT_QUERIES` | `2` |
| LLM 模型 | `.env` `DEEPSEEK_MODEL` | `deepseek-chat` |

---

## 高级用法

### Human-in-the-loop（HITL）

```bash
uv run python main.py "你的问题" --interactive
# 在每次 quality_eval 后暂停，5 选 1：approve / reject / force_final / edit_report / custom_score
```

### RAG（本地知识库）

```bash
# 1. 把文档放到 data/knowledge/（支持 .md / .txt / .pdf）
# 2. 入库（首次会下载 BGE-small-zh-v1.5 ~120MB）
uv run python scripts/ingest_knowledge.py --reset
# 3. 正常跑 main.py，ReAct 会自动选用 local_knowledge_search
```

### 批量评估（eval）

```bash
# 跑 data/eval/questions.json 里的 5 道题，出 markdown 报告
uv run python scripts/run_eval.py

# 切换不同配置对比
RESEARCHER_MODE=simple uv run python scripts/run_eval.py
ENABLE_RAG=false      uv run python scripts/run_eval.py
```

报告保存到 `outputs/eval_<timestamp>.md`，含：综合统计 / 每题明细 / 关键词命中 / 工具使用 / judge feedback。

## 开发

```bash
# 非联网测试（默认）
uv run pytest -m "not live"

# 真实联网集成测试（消耗 token）
uv run pytest -m live

# 代码风格
uv run ruff check src/ main.py scripts/ tests/
```

测试组织：
- `test_*_agent.py` / `test_tools.py` / `test_rag.py` / `test_eval.py` —— 节点/模块级 mock 单测
- `test_graph_dryrun.py` / `test_react_graph_dryrun.py` / `test_human_review.py` —— 主图 e2e 干跑（mock LLM/搜索）
- `test_integration.py` —— 真实联网 e2e（全部 `@pytest.mark.live`）

## 学习笔记

详见 `docs/learning_notes.md`，按 Phase 6 改造路径组织：
- 第 1 章 ReAct / Function Calling
- 第 2 章 RAG
- 第 3 章 Human-in-the-loop
- 第 4 章 Evaluation

---

## 关键设计偏差（与原 spec 的差异）

| 原 spec | 实际实现 | 原因 |
|---|---|---|
| Tavily 搜索 | DuckDuckGo (`ddgs`) | 零配置、免 key |
| Researcher 用 `create_react_agent` | 确定性链路（关键词抽取 → 多查询 → 压缩） | 单工具下 ReAct 无价值，节省 LLM 调用 |
| Supervisor 在 `next_agent` 路由时 `iteration_count++` | 仅由 `revision_node` 维护 | 让计数语义=自进化轮数，避免首次跑就 +5 |
| 只有 `draft_writer` 节点 | 拆出 `brief_writer` 单独挂主图 | 否则 `researcher` 拿不到 `research_brief` |
| Quality 评分 JSON 解析失败 → 默认 5.0 | 同 + 字段范围裁剪 0~10 + 原始返回写入 feedback | 防呆 + 便于排查 |

---

## 目录结构

```
deep-research-agent/
├── main.py                 # CLI 入口
├── scripts/
│   └── verify_env.py       # Phase 0 环境黑箱探测
├── src/
│   ├── config.py           # .env → settings + get_llm()
│   ├── state.py            # SupervisorState / QualityScore TypedDict
│   ├── graph.py            # 主图组装
│   ├── tools/
│   │   ├── search_tool.py  # DDG 封装
│   │   └── compress_tool.py
│   └── agents/
│       ├── supervisor_agent.py
│       ├── draft_agent.py
│       ├── researcher_agent.py
│       ├── red_team_agent.py
│       ├── quality_agent.py
│       ├── revision_agent.py
│       └── final_report_agent.py
├── tests/                  # 单测 + 干跑 + 集成
├── outputs/                # 生成的报告
├── pyproject.toml          # uv 项目（清华 PyPI 镜像）
├── pytest.ini              # pytest + live marker
└── .env / .env.example
```
