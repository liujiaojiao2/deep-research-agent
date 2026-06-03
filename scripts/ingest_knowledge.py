"""Knowledge base 入库脚本。

用法：
    uv run python scripts/ingest_knowledge.py [--reset] [--dir DIR] [--include-images]

首次运行会下载 BGE-small-zh-v1.5 (~120MB)；
--include-images 时还会下载 Chinese-CLIP (~600MB)。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console

from src.rag import (  # noqa: E402
    KNOWLEDGE_DIR,
    add_image_to_kb,
    ingest_directory,
)

console = Console()

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def scan_images(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def main():
    parser = argparse.ArgumentParser(description="Ingest knowledge base for RAG")
    parser.add_argument("--reset", action="store_true", help="reset collection 后再入库")
    parser.add_argument("--dir", default=None, help="覆盖 knowledge 目录路径")
    parser.add_argument(
        "--include-images", action="store_true",
        help="同时用 Chinese-CLIP 把 PNG/JPG 图片嵌入到图片库"
    )
    args = parser.parse_args()

    root = Path(args.dir) if args.dir else KNOWLEDGE_DIR
    console.rule(f"[bold cyan]📚 Ingesting knowledge from {root}[/bold cyan]")
    console.print("[dim]首次运行会下载嵌入模型（文本 ~120MB）...[/dim]")

    # ── 文本入库 ──
    stats = ingest_directory(root=root, reset=args.reset)
    text_ok = stats["chunks"] > 0

    # ── 图片入库 ──
    image_count = 0
    if args.include_images:
        console.print("[dim]加载 Chinese-CLIP 图片嵌入模型（首次 ~600MB）...[/dim]")
        images = scan_images(root)
        for img_path in images:
            rel = str(img_path.relative_to(root))
            ok = add_image_to_kb(str(img_path), source_label=rel)
            if ok:
                image_count += 1
                console.print(f"  🖼️  [dim]{rel}[/dim]")

    console.rule()
    if text_ok:
        console.print(f"[green]✅ 文本入库[/green]  collection=[bold]{stats['collection']}[/bold]  "
                      f"文件={stats['files']}  切块={stats['chunks']}")
    else:
        console.print("[yellow]⚠ 没有可入库的文本文件（.md/.txt/.pdf）[/yellow]")

    if args.include_images:
        console.print(f"🖼️  图片入库: [bold]{image_count}[/bold] 张")
    else:
        console.print("[dim]💡 加 --include-images 可同时入库图片 (Chinese-CLIP 多模态嵌入)[/dim]")


if __name__ == "__main__":
    main()
