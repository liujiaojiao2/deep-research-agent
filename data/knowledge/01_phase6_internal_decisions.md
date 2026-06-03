# Phase 6 内部技术决策记录（虚构 demo 文档 · 用于 RAG 检索测试）

> 本文档为 DeepResearch Agent 项目 Phase 6 阶段的内部决策记录，记录人：刘九原（虚构），日期：2026-05-15。
> 注：本文档为 RAG 演示用虚构内容，LLM 预训练知识里不存在这些信息。

## 1. ReAct 工具集裁定

经 2026-05-12 内部讨论，Phase 6.1 的 ReAct researcher 工具集最终确定为以下 5 类：

| 工具名 | 适用场景 | 调用配额（单次研究） |
|---|---|---|
| web_search (DuckDuckGo) | 时效性强的新闻、博客 | 不限 |
| wikipedia_search | 概念定义、历史背景 | 上限 3 次 |
| arxiv_search | 学术论文 | 上限 5 次 |
| python_calculator | 数值计算 | 不限 |
| get_current_datetime | 时效判断 | 通常 1 次 |

否决方案：原计划接入 **Google Scholar**，因为爬虫频率限制太严苛被否；备选方案 **Semantic Scholar API** 留作 Phase 6.4 候选。

## 2. RESEARCHER_MODE 默认值之争

围绕"默认走 ReAct 还是 simple"，团队有两派意见：

- **A 派（默认 simple）**：理由是 token 成本低、确定性高，更适合面向客户演示
- **B 派（默认 react，最终选项）**：理由是体现 Agent 自主性，长期是主线路径

最终拍板：默认 `RESEARCHER_MODE=react`，但 README 与 .env.example 里都明确说明可切换；测试 fixture 锁定 simple 路径，避免每次回归都消耗真实 LLM。

## 3. 内部基准数据

我们用 30 道研究题做了一组对比测试（2026-05-13 凌晨跑完）：

| 模式 | 平均 token 消耗 | 平均完成时间 | 平均 Quality overall | citation 平均分 |
|---|---|---|---|---|
| simple | 4,200 | 38 秒 | 8.31 | 7.10 |
| react（不强制工具） | 3,100 | 22 秒 | 8.07 | 1.95 |
| react（强制多工具） | 11,800 | 71 秒 | 8.62 | 7.85 |

**关键洞察**：ReAct 模式如果不在 prompt 里"硬约束"，LLM 会偷懒不调工具，citation 分会暴跌。这就是为什么 `REACT_SYSTEM_PROMPT` 第 4 条写明"不要在没调工具的情况下凭知识回答"。

## 4. 与 Phase 6.2 的接口约定

Phase 6.2（RAG）会在 ReAct 工具集首位插入 `local_knowledge_search`。约定：

- 本地知识库优先级最高（LLM 应先查本地再查 web）
- 空库时工具应**友好返回**「本地知识库为空，请改用其他工具」，而不是抛异常
- ChromaDB 持久化目录：`.chroma/`（已加入 `.gitignore`）
- 默认 collection 名：`deep_research_kb`

## 5. 已知风险

- ReAct 递归上限：当前默认 25。曾观察到 LLM 在补搜索阶段连续调 12 次工具仍未停止——已加 try/except 兜底。
- DuckDuckGo 限流：每分钟超过 20 次会被限流。Phase 6.4 计划接入备用搜索（Brave Search API）。
- DeepSeek Function Calling 在中文复杂场景偶发 JSON 输出失败率约 3%，已在 quality_agent 加容错。
