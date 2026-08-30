"""从 Markdown 报告抽取大纲和思维导图 HTML。

设计要点：
- **单一真源**：从最终报告的 heading + bullet 层级抽出规范化 Outline
  （`# / ## / -`），outline 本身即"大纲视图"；用 markmap-autoloader
  把同一份 outline 渲染成可缩放/折叠的交互式思维导图 HTML。
- **纯正则骨架**：`report_to_skeleton()` 无 LLM 依赖、无网络，
  确定性可测，任何情况下都是可靠兜底。
- **可选 LLM 叶子增强**（Phase B）：`report_to_outline(enrich=True)`
  会为每个 heading 补充 3-5 个从原文提炼的关键要点作为子 bullet。
  失败自动降级为纯骨架。
- CDN 依赖：mindmap.html 依赖 `cdn.jsdelivr.net` 上的 markmap-autoloader；
  air-gapped 环境下 mindmap.html 无法渲染，但 outline.md 仍是可用文本输出。

CLI：
    uv run python -m src.tools.mindmap_tool <path/to/report.md> [--enrich]
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import logging
import re
import sys
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_EXCEPTION
from pathlib import Path

logger = logging.getLogger(__name__)

_MARKMAP_CDN = "https://cdn.jsdelivr.net/npm/markmap-autoloader@0.18"

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
_BULLET_RE = re.compile(r"^(\s*)[-*+]\s+(.+)$")
_FENCE_RE = re.compile(r"^(```|~~~)")


# ---------- 内联清理 ----------

def _clean_inline(text: str) -> str:
    """剥离 heading/bullet 文本里的常见内联 markdown 标记。

    - `**bold**` / `__bold__` → 内容
    - `` `code` `` → 内容
    - `[text](url)` → text
    - 保留 CJK 等一切其它字符
    """
    text = text.strip()
    # 链接优先：[text](url) → text
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
    # 加粗 / 斜体强调
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"__(.+?)__", r"\1", text)
    # 行内代码
    text = re.sub(r"`([^`]+)`", r"\1", text)
    return text.strip()


# ---------- Skeleton 抽取 ----------

def report_to_skeleton(md: str, *, max_depth: int = 6) -> str:
    """从 Markdown 报告抽取规范化 outline 骨架（headings + bullets）。

    纯正则实现，无 LLM。逐行扫描，跳过 fenced code block 内部。
    - Heading `# / ## / ...` 直接映射为 outline 同级 heading。
    - Bullet 深度 = heading level + 1 + 缩进深度（每 2 空格算 1 级），
      被 `max_depth` 截断。
    - 无 heading 时合成 `# Report` 根，把 bullets 挂上。
    - 完全空 → 返回 `# (empty report)`。
    """
    lines = (md or "").splitlines()
    out_lines: list[str] = []
    in_fence = False
    current_heading_level = 0  # 尚未见 heading 之前 = 0

    has_any_heading = False
    pending_bullets_before_first_heading: list[tuple[int, str]] = []

    for raw in lines:
        # fenced code block 状态机
        if _FENCE_RE.match(raw.strip()):
            in_fence = not in_fence
            continue
        if in_fence:
            continue

        # heading
        m_h = _HEADING_RE.match(raw)
        if m_h:
            level = min(len(m_h.group(1)), max_depth)
            text = _clean_inline(m_h.group(2))
            if not text:
                continue
            has_any_heading = True
            # 如果之前有 orphan bullets，需要在此时先补一个合成 heading
            if pending_bullets_before_first_heading and not out_lines:
                out_lines.append("# Report")
                for depth, btext in pending_bullets_before_first_heading:
                    d = min(depth, max_depth - 1)
                    out_lines.append(("  " * d) + "- " + btext)
                pending_bullets_before_first_heading.clear()
            current_heading_level = level
            out_lines.append(("#" * level) + " " + text)
            continue

        # bullet
        m_b = _BULLET_RE.match(raw)
        if m_b:
            indent_spaces = len(m_b.group(1).expandtabs(4))
            bullet_depth = indent_spaces // 2  # 缩进→深度
            text = _clean_inline(m_b.group(2))
            if not text:
                continue
            if current_heading_level == 0:
                # 记录到 pending，等看到第一个 heading 再决定；
                # 若始终没有 heading，则合成根 heading
                pending_bullets_before_first_heading.append((bullet_depth + 1, text))
                continue
            depth = current_heading_level + 1 + bullet_depth
            depth = min(depth, max_depth)
            out_lines.append(("  " * (depth - 1)) + "- " + text)
            continue

        # 其它行忽略（正文段落不进骨架）

    # 全文无 heading 但有 bullets：合成根
    if not has_any_heading and pending_bullets_before_first_heading:
        out_lines.append("# Report")
        for depth, btext in pending_bullets_before_first_heading:
            d = min(depth, max_depth - 1)
            out_lines.append(("  " * d) + "- " + btext)

    if not out_lines:
        return "# (empty report)"

    return "\n".join(out_lines)


# ---------- 顶层 Outline 入口 ----------

def report_to_outline(
    md: str,
    *,
    enrich: bool = False,
    max_depth: int = 6,
    llm=None,
) -> str:
    """从报告抽 outline；enrich=True 时启用 LLM 叶子增强。

    enrich 内部任何异常都会捕获，降级为纯骨架 —— 显示层功能不能拖垮流程。
    """
    skeleton = report_to_skeleton(md, max_depth=max_depth)
    if not enrich:
        return skeleton
    try:
        return enrich_with_leaves(skeleton, md, llm=llm, max_depth=max_depth)
    except Exception as e:
        logger.warning("enrich_with_leaves failed, fall back to skeleton: %s", e)
        return skeleton


# ---------- Enrich：LLM 叶子增强 ----------

def _split_sections(md: str) -> list[tuple[tuple[str, ...], str]]:
    """把 Markdown 按 heading 切分成 [(heading_path, body_text), ...]。

    - heading_path 是从根到当前 heading 的清理后文本元组。
    - body_text 是该 heading 之后、下一个同级/更高级 heading 之前的正文，
      去除 fenced code block 内部。
    - 无 heading 时返回空列表。
    """
    lines = (md or "").splitlines()
    sections: list[tuple[tuple[str, ...], list[str]]] = []
    stack: list[tuple[int, str]] = []  # (level, cleaned_text)
    current_body: list[str] = []
    current_path: tuple[str, ...] | None = None
    in_fence = False

    def _flush():
        nonlocal current_body, current_path
        if current_path is not None:
            sections.append((current_path, list(current_body)))
        current_body = []

    for raw in lines:
        if _FENCE_RE.match(raw.strip()):
            in_fence = not in_fence
            if current_path is not None:
                current_body.append(raw)
            continue
        if in_fence:
            if current_path is not None:
                current_body.append(raw)
            continue

        m_h = _HEADING_RE.match(raw)
        if m_h:
            _flush()
            level = len(m_h.group(1))
            text = _clean_inline(m_h.group(2))
            if not text:
                current_path = None
                continue
            # 弹掉栈里 level >= 当前 level 的
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, text))
            current_path = tuple(t for _, t in stack)
            continue

        if current_path is not None:
            current_body.append(raw)

    _flush()
    # body 转字符串，剥 fenced code 后的内容
    result: list[tuple[tuple[str, ...], str]] = []
    for path, body_lines in sections:
        # 移除 fenced code block 内部行（保留 heading 上下文即可）
        clean_body: list[str] = []
        in_fence_local = False
        for l in body_lines:
            if _FENCE_RE.match(l.strip()):
                in_fence_local = not in_fence_local
                continue
            if in_fence_local:
                continue
            clean_body.append(l)
        body = "\n".join(clean_body).strip()
        if body:
            result.append((path, body))
    return result


def _extract_leaves_from_llm(
    llm,
    heading: str,
    body: str,
    *,
    max_bullets: int = 5,
    body_char_limit: int = 2000,
) -> list[str]:
    """调 LLM 从段落里抽 3-5 个不超过 30 字的关键要点。

    返回 list[str]。任何失败 → 返回 []（该 section 无叶子，不阻断）。
    """
    body_snip = body[:body_char_limit]
    prompt = (
        "你是一个大纲提炼助手。请从下面段落中提取 3-5 个关键要点，"
        "每点不超过 30 字。要求：\n"
        "1. 只使用文本中出现的内容，禁止编造或引申。\n"
        "2. 输出严格的 JSON 数组，形如 [\"要点1\", \"要点2\", ...]。\n"
        "3. 不要输出任何解释、不要用代码围栏。\n\n"
        f"段落（属于 heading: {heading}）:\n{body_snip}"
    )
    try:
        resp = llm.invoke(prompt)
        content = resp.content if hasattr(resp, "content") else str(resp)
    except Exception as e:
        logger.debug("enrich LLM invoke failed on %r: %s", heading, e)
        return []

    # 从返回里抽出第一个 JSON 数组
    m = re.search(r"\[[\s\S]*?\]", content)
    if not m:
        return []
    try:
        arr = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    if not isinstance(arr, list):
        return []
    leaves: list[str] = []
    for item in arr[:max_bullets]:
        if isinstance(item, str) and item.strip():
            leaves.append(item.strip())
    return leaves


# 内存缓存：Streamlit 反复渲染同一报告时避免重复 LLM 调用
_ENRICH_CACHE: dict[tuple[str, str], list[str]] = {}


def _cache_key(heading_path: tuple[str, ...], body: str) -> tuple[str, str]:
    body_hash = hashlib.sha1(body.encode("utf-8", errors="replace")).hexdigest()
    return (" > ".join(heading_path), body_hash)


def enrich_with_leaves(
    skeleton: str,
    md: str,
    *,
    llm=None,
    max_bullets: int = 5,
    max_workers: int = 4,
    max_depth: int = 6,
    per_section_timeout: float = 20.0,
    total_timeout: float = 60.0,
) -> str:
    """在 skeleton 每个 heading 后追加从原文段落提炼的叶子 bullet。

    - `llm=None` 时用 `model_router.get_llm_for("memory")`。
    - 并发调 LLM（`ThreadPoolExecutor(max_workers)`），单 section 超时或失败
      降级为无叶子（该 heading 仅保留骨架），不影响其他 section。
    - 结果按 skeleton 中 heading 出现顺序穿插回去。
    """
    sections = _split_sections(md)
    if not sections:
        return skeleton

    if llm is None:
        # 延迟 import：mock LLM 测试时不触发 model_router 加载 config
        from src.model_router import get_llm_for
        llm = get_llm_for("memory")

    # 并发抽取
    path_to_leaves: dict[tuple[str, ...], list[str]] = {}

    def _job(path, body):
        key = _cache_key(path, body)
        if key in _ENRICH_CACHE:
            return path, _ENRICH_CACHE[key]
        leaves = _extract_leaves_from_llm(
            llm, path[-1], body, max_bullets=max_bullets
        )
        _ENRICH_CACHE[key] = leaves
        return path, leaves

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_job, p, b) for p, b in sections]
        done, not_done = wait(futures, timeout=total_timeout)
        for fut in done:
            try:
                path, leaves = fut.result(timeout=per_section_timeout)
                path_to_leaves[path] = leaves
            except Exception as e:
                logger.debug("enrich section failed: %s", e)
        for fut in not_done:
            fut.cancel()

    # 把 leaves 穿插进 skeleton
    skeleton_lines = skeleton.splitlines()
    out: list[str] = []
    # 建 heading_path 追踪，用 heading text 匹配（skeleton 里 heading 顺序 = md 里顺序）
    heading_stack: list[tuple[int, str]] = []
    for line in skeleton_lines:
        out.append(line)
        m_h = _HEADING_RE.match(line)
        if not m_h:
            continue
        level = len(m_h.group(1))
        text = m_h.group(2).strip()
        while heading_stack and heading_stack[-1][0] >= level:
            heading_stack.pop()
        heading_stack.append((level, text))
        path = tuple(t for _, t in heading_stack)
        leaves = path_to_leaves.get(path, [])
        if not leaves:
            continue
        depth = min(level + 1, max_depth)
        indent = "  " * (depth - 1)
        for leaf in leaves:
            out.append(f"{indent}- {leaf}")

    return "\n".join(out)


# ---------- HTML 渲染 ----------

def outline_to_markmap_html(outline: str, title: str = "MindMap") -> str:
    """把 outline 嵌入 markmap-autoloader HTML 模板。

    outline 位于 `<script type="text/template">` 内，不被浏览器当 HTML 解析；
    唯一的防御性处理是把可能出现的 `</script` 替换为 `<\\/script`。
    """
    safe_outline = outline.replace("</script", "<\\/script")
    safe_title = html.escape(title or "MindMap")
    return (
        "<!doctype html><html><head><meta charset=\"utf-8\">"
        f"<title>{safe_title}</title>"
        "<style>html,body,#mm{height:100%;margin:0;"
        "font-family:-apple-system,BlinkMacSystemFont,sans-serif}</style>"
        "</head><body>"
        "<svg id=\"mm\"></svg>"
        f"<script src=\"{_MARKMAP_CDN}\"></script>"
        "<script type=\"text/template\" class=\"markmap\" data-mm-svg=\"#mm\">\n"
        f"{safe_outline}\n"
        "</script>"
        "</body></html>"
    )


def report_to_mindmap_html(
    md: str,
    *,
    enrich: bool = False,
    title: str = "MindMap",
    llm=None,
) -> str:
    """组合便捷：报告 → outline → mindmap HTML。"""
    outline = report_to_outline(md, enrich=enrich, llm=llm)
    return outline_to_markmap_html(outline, title=title)


# ---------- 落盘：兄弟文件 ----------

def write_siblings(
    md_path: Path,
    md_text: str,
    *,
    enrich: bool = False,
    title: str = "",
    llm=None,
) -> tuple[Path, Path]:
    """在 `md_path` 旁边写 `.outline.md` 与 `.mindmap.html`，返回两条路径。"""
    md_path = Path(md_path)
    stem = md_path.with_suffix("")  # 保留全路径，去 .md 后缀
    outline_path = stem.with_suffix(".outline.md")
    mm_path = stem.with_suffix(".mindmap.html")

    outline = report_to_outline(md_text, enrich=enrich, llm=llm)
    mm_html = outline_to_markmap_html(outline, title=title or md_path.stem)

    outline_path.write_text(outline, encoding="utf-8")
    mm_path.write_text(mm_html, encoding="utf-8")
    return outline_path, mm_path


# ---------- CLI ----------

def _cli(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="mindmap_tool",
        description="Convert a Markdown report to outline + interactive mindmap HTML.",
    )
    p.add_argument("md_path", help="Path to a Markdown report file")
    p.add_argument("--title", default="", help="Title for the mindmap HTML (default: file stem)")
    p.add_argument(
        "--enrich",
        action="store_true",
        help="Enrich each heading with 3-5 key bullets via LLM (extra token cost)",
    )
    args = p.parse_args(argv)

    md_path = Path(args.md_path)
    if not md_path.is_file():
        print(f"error: file not found: {md_path}", file=sys.stderr)
        return 2

    md_text = md_path.read_text(encoding="utf-8")
    outline_path, mm_path = write_siblings(
        md_path, md_text, enrich=args.enrich, title=args.title
    )
    print(f"outline: {outline_path}")
    print(f"mindmap: {mm_path}")
    return 0


if __name__ == "__main__":
    sys.exit(_cli())
