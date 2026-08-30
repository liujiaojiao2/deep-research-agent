"""mindmap_tool 骨架 + 增强单测（全离线）。

Phase A：12 个骨架用例，无 LLM 依赖。
Phase B：5 个 enrich 用例，用 mock LLM 注入。
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from src.tools.mindmap_tool import (
    _ENRICH_CACHE,
    _cli,
    _split_sections,
    enrich_with_leaves,
    outline_to_markmap_html,
    report_to_mindmap_html,
    report_to_outline,
    report_to_skeleton,
    write_siblings,
)


@pytest.fixture(autouse=True)
def _clear_enrich_cache():
    """避免测试之间 cache 串扰。"""
    _ENRICH_CACHE.clear()
    yield
    _ENRICH_CACHE.clear()


# ---------- Phase A：骨架 12 用例 ----------

def test_headings_only():
    md = "# 一级\n\n## 二级\n\n### 三级\n"
    out = report_to_skeleton(md)
    assert out.splitlines() == ["# 一级", "## 二级", "### 三级"]


def test_nested_bullets():
    md = (
        "## H2 章节\n"
        "- a\n"
        "  - b\n"
        "    - c\n"
    )
    out = report_to_skeleton(md).splitlines()
    # H2 = level 2；bullets 深度 = 2 + 1 + [0,1,2] = 3,4,5 → 前缀 2,3,4 个双空
    assert out[0] == "## H2 章节"
    assert out[1] == "    - a"       # 深度 3 → 2 层缩进
    assert out[2] == "      - b"     # 深度 4 → 3 层缩进
    assert out[3] == "        - c"   # 深度 5 → 4 层缩进


def test_empty_report():
    out = report_to_skeleton("")
    assert out == "# (empty report)"
    html = outline_to_markmap_html(out)
    assert "<script" in html and "</script>" in html
    assert "(empty report)" in html


def test_code_block_hashes_ignored():
    md = (
        "# 真 heading\n"
        "```python\n"
        "# not a heading\n"
        "## also not\n"
        "```\n"
        "## 真二级\n"
    )
    out = report_to_skeleton(md).splitlines()
    assert out == ["# 真 heading", "## 真二级"]


def test_cjk_characters():
    md = "## 多头注意力机制\n"
    assert report_to_skeleton(md).strip() == "## 多头注意力机制"


def test_bold_and_link_inline_stripped():
    md = "## **Bold** and [Link](http://example.com)\n"
    assert report_to_skeleton(md).strip() == "## Bold and Link"


def test_html_contains_outline_verbatim():
    outline = "# Root\n## Sub\n"
    html = outline_to_markmap_html(outline, title="T")
    # outline 逐字保留在 script template 内
    assert outline in html
    assert 'class="markmap"' in html


def test_html_is_self_contained():
    html = outline_to_markmap_html("# X\n")
    assert "cdn.jsdelivr.net" in html
    assert "markmap-autoloader" in html
    # 无本地文件引用
    assert "file://" not in html
    assert "./" not in html


def test_no_headings_only_bullets():
    md = "- 只有 bullet\n- 第二个\n"
    out = report_to_skeleton(md).splitlines()
    assert out[0] == "# Report"
    assert "- 只有 bullet" in out[1]
    assert "- 第二个" in out[2]


def test_deep_nesting_clamped():
    # 10 层缩进（每 2 空格一层），H1 之下 → 深度 = 1 + 1 + 10 = 12
    md = "# 根\n" + "                    - 很深\n"  # 20 空格 = 10 层
    out = report_to_skeleton(md, max_depth=6).splitlines()
    # 深度截断到 6 → 5 层缩进 = 10 空格
    assert out[0] == "# 根"
    assert out[1] == "          - 很深"


def test_write_siblings_paths(tmp_path: Path):
    md_path = tmp_path / "report_20260101.md"
    md_path.write_text("# 标题\n## 子标题\n", encoding="utf-8")
    outline_p, mm_p = write_siblings(md_path, md_path.read_text(encoding="utf-8"))
    assert outline_p == tmp_path / "report_20260101.outline.md"
    assert mm_p == tmp_path / "report_20260101.mindmap.html"
    assert outline_p.exists()
    assert mm_p.exists()
    assert "# 标题" in outline_p.read_text(encoding="utf-8")
    assert "markmap-autoloader" in mm_p.read_text(encoding="utf-8")


def test_cli_invocation(tmp_path: Path):
    md_path = tmp_path / "r.md"
    md_path.write_text("# 一级\n- 要点\n", encoding="utf-8")
    rc = _cli([str(md_path)])
    assert rc == 0
    assert (tmp_path / "r.outline.md").exists()
    assert (tmp_path / "r.mindmap.html").exists()


# ---------- Phase B：enrich 5 用例 ----------


class _StubLLM:
    """按 heading text 决定回复的 mock LLM。

    - responses: {heading_text_or_None: str}；None 是通配（默认回复）
    - raise_on: set[str] 里的 heading 会让 invoke 抛异常
    - call_log: 收集所有实际调用的 heading
    """

    def __init__(self, responses=None, raise_on=None):
        self.responses = responses or {}
        self.raise_on = set(raise_on or [])
        self.call_log: list[str] = []

    def invoke(self, prompt: str):
        # 从 prompt 里解析出 heading 名
        m = re.search(r"heading: (.+?)\)", prompt)
        heading = m.group(1) if m else ""
        self.call_log.append(heading)
        if heading in self.raise_on:
            raise RuntimeError(f"boom on {heading}")
        content = self.responses.get(heading, self.responses.get(None, "[]"))
        return SimpleNamespace(content=content)


import re  # for _StubLLM


def test_split_sections():
    md = (
        "# 一级\n"
        "一级正文\n"
        "## 二级 A\n"
        "二级 A 正文\n"
        "### 三级\n"
        "三级正文\n"
        "## 二级 B\n"
        "二级 B 正文\n"
    )
    sections = _split_sections(md)
    paths = [p for p, _ in sections]
    assert paths == [
        ("一级",),
        ("一级", "二级 A"),
        ("一级", "二级 A", "三级"),
        ("一级", "二级 B"),
    ]
    bodies = {p: body for p, body in sections}
    assert "一级正文" in bodies[("一级",)]
    assert "二级 A 正文" in bodies[("一级", "二级 A")]
    assert "三级正文" in bodies[("一级", "二级 A", "三级")]
    assert "二级 B 正文" in bodies[("一级", "二级 B")]
    # heading 内容不串到父 section
    assert "二级 A 正文" not in bodies[("一级",)]


def test_enrich_appends_leaves():
    md = (
        "# 根\n"
        "根正文\n"
        "## A\n"
        "A 正文\n"
        "## B\n"
        "B 正文\n"
    )
    llm = _StubLLM(responses={None: '["点1", "点2", "点3"]'})
    out = enrich_with_leaves(report_to_skeleton(md), md, llm=llm, max_workers=2)
    lines = out.splitlines()
    # 根/A/B 三个 heading 各自后面都有 3 条叶子
    assert lines[0] == "# 根"
    assert lines[1] == "  - 点1"
    assert lines[2] == "  - 点2"
    assert lines[3] == "  - 点3"
    assert "## A" in lines
    a_idx = lines.index("## A")
    assert lines[a_idx + 1] == "    - 点1"  # H2 → 深度 3 → 2 层缩进
    # 共 3 个 heading × 3 leaves + 3 heading = 12 行
    assert len(lines) == 12


def test_enrich_llm_failure_falls_back():
    md = "# X\nX 正文\n## Y\nY 正文\n"
    llm = _StubLLM(
        responses={None: '["ok1", "ok2"]'},
        raise_on={"X"},  # X 失败，Y 正常
    )
    out = enrich_with_leaves(report_to_skeleton(md), md, llm=llm, max_workers=2)
    lines = out.splitlines()
    # X 保留 heading，无叶子
    assert lines[0] == "# X"
    # Y 有叶子
    y_idx = lines.index("## Y")
    assert lines[y_idx + 1] == "    - ok1"


def test_enrich_json_parse_failure():
    md = "# X\nX 正文\n"
    llm = _StubLLM(responses={None: "这不是 JSON，我瞎聊聊"})
    out = enrich_with_leaves(report_to_skeleton(md), md, llm=llm)
    # 该 section 无叶子，不抛
    assert out.strip() == "# X"


def test_enrich_disabled_matches_skeleton():
    md = "# A\n## B\n段落文本\n- 点\n"
    assert report_to_outline(md, enrich=False) == report_to_skeleton(md)


def test_enrich_top_level_exception_falls_back_to_skeleton(monkeypatch):
    """report_to_outline(enrich=True) 内部异常兜底：整体退回骨架。"""
    md = "# A\n段落\n"

    def _boom(*a, **k):
        raise RuntimeError("total meltdown")

    monkeypatch.setattr("src.tools.mindmap_tool.enrich_with_leaves", _boom)
    out = report_to_outline(md, enrich=True, llm=_StubLLM())
    assert out == report_to_skeleton(md)


# ---------- 组合便捷入口 ----------

def test_report_to_mindmap_html_composes():
    md = "# X\n## Y\n"
    html = report_to_mindmap_html(md, title="hello")
    assert "hello" in html
    assert "# X" in html
    assert "## Y" in html
