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
│ supervisor  │◀──────────────────────────────────┐
└──────┬──────┘                                   │
       │ 条件路由（纯函数状态机，零 token）          │
       ├──▶ brief_writer ──────────────────────────┤
       ├──▶ researcher (ReAct/ReWOO + 多源搜索) ──┤
       ├──▶ draft_writer ──────────────────────────┤
       ├──▶ quality_eval (5 维 JSON 评分) ────────┤
       ├──▶ red_team → revision ───────────────────┘
       └──▶ final_report → memory_archive
                              → evolution_log
                              → skill_library → END
```

Supervisor 决策表（按优先级短路）：

| 状态 | 路由 |
|---|---|
| 无 `research_brief` | `brief_writer` |
| 无 `research_results` | `researcher` |
| 无 `draft_report` | `draft_writer` |
| 无 `quality_score` | `quality_eval` |
| `overall >= QUALITY_THRESHOLD` | `final_report` |
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

## 关键设计决策（ADR）

> ADR = Architecture Decision Record。记录关键架构决策的背景、备选方案、选择理由和后果。

### 1. Supervisor 状态机 vs 链式 DAG

**背景：** 多 Agent 系统需要协调 6+ 个节点按正确的顺序执行，且需要支持条件分支（低分走 red_team，高分直接结稿）。

**备选方案：**
| 方案 | 优点 | 缺点 |
|------|------|------|
| 固定链式 DAG | 简单直观 | 不支持条件分支，自进化无法实现 |
| 纯 LLM 路由 | 灵活 | 每次路由消耗 token，不稳定 |
| **纯函数状态机** ✅ | 零 token 消耗，确定性高 | 新增节点需手动更新决策表 |

**决策：** 用纯函数状态机（`supervisor_node`），按优先级短路决策。每次只写 `next_agent` 一个字段。

**后果：** 
- ✅ 路由决策零 token 成本、零延迟
- ✅ 行为可预测，测试可直接验证路由表
- ❌ 新增 Agent 时需要手动更新决策树（维护成本 O(1)）

---

### 2. ReWOO vs ReAct vs Plan-and-Execute

**背景：** Researcher 节点负责多源搜索（Web / Wikipedia / ArXiv / 本地知识库）。ReAct 模式下每步工具调用 = 1 次 LLM 推理，N 步 = N+1 次 LLM 调用。

**备选方案：**
| 方案 | LLM 调用 | 工具调用 | 适用场景 |
|------|----------|----------|---------|
| ReAct | N+1 次 | N 次 | 工具不可预测、需动态决策 |
| Plan-and-Execute | 2 次 | N 次 | 工具可预定义、需 replan |
| **ReWOO** ✅ | **1 次** | N 次 | 工具可预定义、一步到位 |

**决策：** 默认 ReAct（兼容性最好），通过 `RESEARCHER_MODE=rewoo` 切换到 ReWOO。ReWOO 把 LLM 调用从 N+1 降到 1 次（规划一次、执行纯函数），实测节省 40-60% token。

**后果：**
- ✅ ReWOO 大幅降低 researcher 节点的 token 消耗
- ❌ ReWOO 无法根据中间结果动态调整策略（对简单/中等复杂度问题足够）
- ✅ 通过环境变量切换，两种模式可对比评估

---

### 3. Red Team + Revision 闭环 vs Self-Consistency 多采样

**背景：** 如何保证 LLM 输出质量？两条主流路线：生成时多采样取中位数（Self-Consistency），或生成后对抗审查+修订（Red Team）。

**备选方案：**
| 方案 | 原理 | LLM 成本 | 适用 |
|------|------|----------|------|
| Self-Consistency | 同一 prompt 跑 N 次取中位数 | N× | 数学/推理类 |
| **Red Team + Revision** ✅ | 对抗审查 → 补搜索 → 重写 | 2-3× | 研究/写作类 |

**决策：** Red Team + Revision 作为主闭环，Self-Consistency 作为 quality_eval 内部的可选增强（`QUALITY_EVAL_SAMPLES > 1` 时多采样评分取中位数）。

**后果：**
- ✅ Red Team 能发现 Self-Consistency 漏掉的系统性幻觉（如编造引用）
- ✅ Revision 补搜索机制弥补信息缺口，而非仅重采样
- ❌ 自进化循环会额外消耗 2-3 次 LLM 调用

---

### 4. 双层评估：内部 quality_eval + 外部 judge

**背景：** 需要知道系统输出好不好，但谁来评？

**备选方案：**
| 方案 | 优点 | 缺点 |
|------|------|------|
| 只用内部 quality_eval | 融入闭环流程 | 自评偏差（实测偏高 0.5-1.0 分） |
| 只用外部 judge | 客观 | 无法融入自进化闭环 |
| **双轨制** ✅ | 闭环+客观兼得 | 多一次 LLM 调用 |

**决策：** quality_eval（内部）融入 Superivsor 路由 → 自进化闭环。独立 judge（外部）用于 eval 批量评估，不与闭环耦合。

**后果：**
- ✅ eval 报告能暴露 quality_eval 的自评偏差
- ✅ 两套 prompt 独立调优，不互相污染
- ❌ `QUALITY_THRESHOLD` 需按自评偏差校准（默认 7.0 对应外部 6.0-6.5）

---

### 5. Prompt Engineering 自进化 vs RL 微调

**背景：** Phase 3 的自进化（red_team → revision）只能改报告文本。如何让系统从历史成功/失败中学习策略？

**备选方案：**
| 方案 | 成本 | 泛化 |
|------|------|------|
| RL 微调 policy model | 高（GPU + 标注数据） | 好 |
| **HarnessForge + Memento-Skills** ✅ | 低（ChromaDB + LLM 提取） | 中 |

**决策：** 用 HarnessForge（归档成功策略 → 同类 query 自动召回）和 Memento-Skills（从高分 run 提取技能模板 → BGE 匹配注入）。不修改模型权重，只优化 agent harness（prompt + tool + config）。

**后果：**
- ✅ 零训练成本，ChromaDB 实现毫秒级检索
- ✅ Agent = Model + Harness，换模型不影响积累的策略
- ❌ 冷启动问题（需要积累一定 run 数后才有召回）

---

### 6. ChromaDB 持久化 vs 纯内存 vs 外接向量库

**背景：** RAG 知识库、演化策略、技能模板、偏好记忆都需要向量存储。

**备选方案：**
| 方案 | 部署复杂度 | 持久化 | 语义检索 |
|------|----------|--------|---------|
| 纯内存 dict | 零 | ❌ | ❌ |
| 外接 Pinecone/Milvus | 高 | ✅ | ✅ |
| **ChromaDB** ✅ | 零（pip install） | ✅ | ✅ |

**决策：** ChromaDB PersistentClient，数据落 `.chroma_db/` 目录。每个 collection 独立（rag / evolution_log / skill_library / memory_preferences）。

**后果：**
- ✅ 零外部依赖，MacBook 本地跑满血
- ✅ 进程退出数据不丢失
- ❌ 不支持分布式（单机 10 万级文档够用）

---

### 7. DeepSeek + LangChain ChatOpenAI vs 多模型路由

**背景：** 模型选择影响成本和质量。不同角色（评分 vs 写作 vs 推理）对模型要求不同。

**备选方案：**
| 方案 | 灵活性 | 复杂度 |
|------|--------|--------|
| 单一模型 | 低 | 低 |
| 自建 multi-provider 路由 | 高 | 高 |
| **DeepSeek + model_router** ✅ | 中 | 中 |

**决策：** 默认 DeepSeek（兼容 OpenAI 协议，通过 `langchain_openai.ChatOpenAI` 调用）。`model_router` 通过环境变量 `MODEL_FOR_<ROLE>` 支持按角色切换模型（如 quality 用 deepseek-reasoner，draft 用 deepseek-chat）。

**后果：**
- ✅ 改模型不改代码（纯环境变量驱动）
- ✅ DeepSeek 性价比高（chat ¥1/2 per 1M IO，对比 GPT-4 ¥70/1M output）
- ❌ 单 provider 依赖（可通过环境变量切换到 OpenAI 兼容的其他 provider）

---

### 8. DuckDuckGo vs Tavily/SerpAPI/Brave

**背景：** Web 搜索是 researcher 的核心工具。

**备选方案：**
| 方案 | 配置 | 质量 | 限制 |
|------|------|------|------|
| Tavily | API key | AI-optimized | 免费 1000/月 |
| SerpAPI | API key | Google 结果 | 免费 100/月 |
| **DuckDuckGo** ✅ | 零配置 | 中 | 无限制 |

**决策：** DuckDuckGo（`ddgs` 库），零配置零 API key。同时支持 Wikipedia、ArXiv、本地知识库作为补充来源。

**后果：**
- ✅ 零成本、零配置、无速率限制
- ✅ 多源互补（DDG + Wikipedia + ArXiv + 本地）
- ❌ 搜索质量不如 Google（对中文/学术内容覆盖一般）

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
