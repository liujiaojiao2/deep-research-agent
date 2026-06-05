"""Obsidian 导出模块 —— 将研究报告一键写入 Obsidian Vault。

Obsidian vault 就是本地 Markdown 文件夹。只需配置 vault 路径即可。

配置：
    .env 中设 OBSIDIAN_VAULT_PATH=/path/to/your/vault
    可选 OBSIDIAN_SUBFOLDER=Research  （vault 内的子目录，默认 Research）
"""
from __future__ import annotations

import logging
import os
import re
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


def _sanitize_filename(name: str, max_len: int = 60) -> str:
    """把 query 转为合法文件名。"""
    safe = re.sub(r'[\\/:*?"<>|]', "", name)
    safe = re.sub(r"\s+", " ", safe).strip()
    return safe[:max_len]


def export_to_obsidian(
    content: str,
    query: str,
    quality_score: dict | None = None,
    tags: list[str] | None = None,
) -> Path | None:
    """将研究报告写入 Obsidian vault。

    Args:
        content: Markdown 报告正文
        query: 研究问题（用作文件名）
        quality_score: 质量评分 dict，写入 frontmatter
        tags: Obsidian tags（不含 #），默认 ["deep-research"]

    Returns:
        写入的文件路径，失败返回 None
    """
    vault_root = os.getenv("OBSIDIAN_VAULT_PATH", "").strip()
    if not vault_root:
        logger.warning("OBSIDIAN_VAULT_PATH 未配置，跳过 Obsidian 导出")
        return None

    vault = Path(vault_root).expanduser()
    if not vault.exists():
        logger.warning("Obsidian vault 路径不存在: %s", vault)
        return None

    subfolder = os.getenv("OBSIDIAN_SUBFOLDER", "Research").strip()
    target_dir = vault / subfolder
    target_dir.mkdir(parents=True, exist_ok=True)

    # 生成文件名：日期_问题.md
    date_str = datetime.now().strftime("%Y%m%d")
    safe_query = _sanitize_filename(query)
    filename = f"{date_str}_{safe_query}.md"
    filepath = target_dir / filename

    # 去重：同名文件加序号
    counter = 1
    while filepath.exists():
        filename = f"{date_str}_{safe_query}_{counter}.md"
        filepath = target_dir / filename
        counter += 1

    # 构建 frontmatter
    tag_list = tags or ["deep-research"]
    tag_str = "\n".join(f"  - {t}" for t in tag_list)
    score_str = ""
    if quality_score:
        overall = quality_score.get("overall", "")
        accuracy = quality_score.get("accuracy", "")
        completeness = quality_score.get("completeness", "")
        score_str = f"""quality_score: {overall}
quality_accuracy: {accuracy}
quality_completeness: {completeness}"""

    frontmatter = f"""---
date: {datetime.now().strftime("%Y-%m-%d")}
tags:
{tag_str}
query: "{query}"
{score_str}
source: DeepResearch Agent
---
"""

    full_content = frontmatter + "\n" + content

    try:
        filepath.write_text(full_content, encoding="utf-8")
        logger.info("Exported to Obsidian: %s", filepath)
        return filepath
    except OSError as e:
        logger.error("Failed to write Obsidian file: %s", e)
        return None


def is_obsidian_configured() -> bool:
    """检查 Obsidian 导出是否已配置。"""
    vault = os.getenv("OBSIDIAN_VAULT_PATH", "").strip()
    return bool(vault) and Path(vault).expanduser().exists()
