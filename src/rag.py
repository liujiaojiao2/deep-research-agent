"""RAG 核心：嵌入 + ChromaDB 持久化 + 检索。

设计要点：
- 单例 collection，名字默认 deep_research_kb（与决策文档约定一致）
- 嵌入模型默认 BGE-small-zh-v1.5；可通过 EMBED_MODEL 环境变量覆盖
- 持久化到项目根目录下 .chroma/（与 .gitignore 对齐）
- ingest 与 retrieve 共用同一份 collection 接入，避免 key 错位
"""
from __future__ import annotations

import hashlib
import logging
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Iterable  # noqa: F401  (运行时未用，但保留类型提示语义)

logger = logging.getLogger(__name__)

DEFAULT_COLLECTION = os.getenv("RAG_COLLECTION", "deep_research_kb")
DEFAULT_EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-zh-v1.5")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHROMA_DIR = PROJECT_ROOT / ".chroma"
KNOWLEDGE_DIR = PROJECT_ROOT / "data" / "knowledge"


# ---------- 嵌入 ----------

@lru_cache(maxsize=1)
def get_embedder():
    """加载 sentence-transformers 嵌入模型；首次会下载 ~120MB。"""
    from sentence_transformers import SentenceTransformer

    logger.info("Loading embedder: %s", DEFAULT_EMBED_MODEL)
    return SentenceTransformer(DEFAULT_EMBED_MODEL)


def embed_texts(texts: list[str]) -> list[list[float]]:
    model = get_embedder()
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return [v.tolist() for v in vectors]


# ---------- 向量库 ----------

# 持有 client 强引用，避免 chromadb 内部 _identifier_to_system 被 GC 后再调时 KeyError
_chroma_client = None


def _build_client():
    import chromadb

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def _force_reset_chroma_internals():
    """绕过 chromadb 内部 SharedSystemClient 的脏状态。"""
    try:
        from chromadb.api.shared_system_client import SharedSystemClient

        SharedSystemClient._identifier_to_system.clear()
        SharedSystemClient._identifier_to_refcount.clear()
    except Exception:
        pass


def get_collection():
    """获取（或创建）持久化的 ChromaDB collection。

    多重防护：
    1. 全局 client 强引用，避免 GC
    2. 第一次失败 → 清 chromadb 内部 SharedSystemClient 字典 → 重建
    3. 仍失败 → 抛异常（让上层捕获）
    """
    global _chroma_client

    def _do():
        global _chroma_client
        if _chroma_client is None:
            _chroma_client = _build_client()
        return _chroma_client.get_or_create_collection(
            name=DEFAULT_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )

    try:
        return _do()
    except (AttributeError, KeyError) as e:
        logger.warning("chromadb 内部状态异常 (%s)，尝试 reset 后重建", e)
        _chroma_client = None
        _force_reset_chroma_internals()
        return _do()


def _reset_client_cache():
    """测试/重置场景用。"""
    global _chroma_client
    _chroma_client = None
    _force_reset_chroma_internals()


# ---------- 切块 ----------

def split_text(text: str, chunk_size: int = 500, overlap: int = 80) -> list[str]:
    """按"段落 → 句号 → 字符"三级切分，保留语义边界。

    保守的实现：先按 "\n\n" 大切块，超长的再用句号 / 换行细切。
    """
    if len(text) <= chunk_size:
        return [text.strip()] if text.strip() else []

    parts: list[str] = []
    paragraphs = re.split(r"\n\s*\n", text)
    buf = ""
    for p in paragraphs:
        p = p.strip()
        if not p:
            continue
        if len(buf) + len(p) + 2 <= chunk_size:
            buf = f"{buf}\n\n{p}" if buf else p
        else:
            if buf:
                parts.append(buf)
            # 单段超长 → 句号细切
            if len(p) > chunk_size:
                sub_parts = _split_long(p, chunk_size, overlap)
                parts.extend(sub_parts[:-1])
                buf = sub_parts[-1]
            else:
                buf = p
    if buf:
        parts.append(buf)
    return [p for p in parts if p.strip()]


def _split_long(text: str, chunk_size: int, overlap: int) -> list[str]:
    """单段超长时按字符 + overlap 切。简单可靠。"""
    parts = []
    i = 0
    while i < len(text):
        parts.append(text[i : i + chunk_size])
        i += chunk_size - overlap
    return parts


# ---------- 入库 ----------

def _read_file(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in (".md", ".txt"):
        return path.read_text(encoding="utf-8")
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError:
            raise ImportError("PDF 支持需要 pypdf，请运行 `uv add pypdf`")
        reader = PdfReader(str(path))
        return "\n\n".join((page.extract_text() or "") for page in reader.pages)
    raise ValueError(f"不支持的文件类型: {suffix}")


def iter_knowledge_files(root: Path = KNOWLEDGE_DIR) -> Iterable[Path]:
    if not root.exists():
        return []
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in (".md", ".txt", ".pdf"))


def ingest_directory(root: Path = KNOWLEDGE_DIR, reset: bool = False) -> dict:
    """把 root 目录下所有支持文档切块嵌入入库。

    reset=True 会先清空 collection。
    返回统计 dict：{"files": N, "chunks": M, "collection": name}。
    """
    coll = get_collection()
    if reset:
        try:
            coll._client.delete_collection(DEFAULT_COLLECTION) if hasattr(coll, "_client") else None
            # 兜底：直接通过全局 client 删
            if _chroma_client is not None:
                try:
                    _chroma_client.delete_collection(DEFAULT_COLLECTION)
                except Exception:
                    pass
        except Exception:
            pass
        coll = get_collection()

    files = list(iter_knowledge_files(root))
    if not files:
        return {"files": 0, "chunks": 0, "collection": DEFAULT_COLLECTION}

    chunks: list[str] = []
    metadatas: list[dict] = []
    ids: list[str] = []

    for f in files:
        text = _read_file(f)
        rel = str(f.relative_to(root))
        for idx, chunk in enumerate(split_text(text)):
            chunks.append(chunk)
            metadatas.append({"source": rel, "chunk_index": idx})
            ids.append(f"{rel}::{idx}")

    if not chunks:
        return {"files": len(files), "chunks": 0, "collection": DEFAULT_COLLECTION}

    embeddings = embed_texts(chunks)
    coll.upsert(ids=ids, documents=chunks, metadatas=metadatas, embeddings=embeddings)
    return {"files": len(files), "chunks": len(chunks), "collection": DEFAULT_COLLECTION}


# ---------- 检索 ----------

def retrieve(query: str, top_k: int = 3) -> list[dict]:
    """语义检索 top_k 个最相关片段。空库时返回 []。"""
    coll = get_collection()
    if coll.count() == 0:
        return []
    embedding = embed_texts([query])[0]
    result = coll.query(query_embeddings=[embedding], n_results=top_k)
    docs = result.get("documents", [[]])[0]
    metas = result.get("metadatas", [[]])[0]
    dists = result.get("distances", [[]])[0]
    return [
        {
            "content": doc,
            "source": meta.get("source", "?"),
            "chunk_index": meta.get("chunk_index", -1),
            "similarity": round(1.0 - float(dist), 4),  # cosine distance → similarity
        }
        for doc, meta, dist in zip(docs, metas, dists)
    ]


# ═══════════════════════════════════════════════════════════════════
# Phase 7.9-extra: 多模态嵌入 (Chinese-CLIP → 图片直接变向量)
# ═══════════════════════════════════════════════════════════════════
# 设计：
#   - 文本嵌入 → BGE (384D) → 原 collection deep_research_kb
#   - 图片嵌入 → Chinese-CLIP (512D) → 新 collection deep_research_kb_images
#   - 检索 → 分别查两个库 → 按 similarity 合并 (图文在同一 CLIP 空间可比)
#   - 开关 ENABLE_IMAGE_RAG=true|false，默认 true

IMAGE_COLLECTION = os.getenv("RAG_IMAGE_COLLECTION", "deep_research_kb_images")
DEFAULT_IMAGE_MODEL = os.getenv("IMAGE_EMBED_MODEL", "OFA-Sys/chinese-clip-vit-base-patch16")

_image_embedder = None


def _get_image_embedder():
    """加载 Chinese-CLIP（首次 ~600MB）；与 BGE 文本嵌入分离。

    优先走离线缓存（HF_HUB_OFFLINE=1），避免代理干扰下载。
    失败时优雅降级：返回 None，让 caller 决定是否回退。
    """
    global _image_embedder
    if _image_embedder is not None:
        return _image_embedder
    try:
        from sentence_transformers import SentenceTransformer

        logger.info("Loading image embedder: %s", DEFAULT_IMAGE_MODEL)
        _image_embedder = SentenceTransformer(DEFAULT_IMAGE_MODEL, local_files_only=True)
    except Exception as e:
        logger.warning("image embedder failed to load (%s), trying with network...", e)
        try:
            from sentence_transformers import SentenceTransformer

            _image_embedder = SentenceTransformer(DEFAULT_IMAGE_MODEL)
        except Exception as e2:
            logger.error("image embedder unavailable: %s", e2)
            _image_embedder = None
    return _image_embedder


def _get_image_collection():
    """获取/创建图片专用 ChromaDB collection（512D，与文本 384D 分库）。"""

    get_collection()  # 触发全局 _chroma_client 初始化
    from src import rag as rag_mod

    def _do():
        return rag_mod._chroma_client.get_or_create_collection(
            name=IMAGE_COLLECTION, metadata={"hnsw:space": "cosine"}
        )

    try:
        return _do()
    except (AttributeError, KeyError) as e:
        logger.warning("image chromadb error (%s) -> reset", e)
        rag_mod._chroma_client = None
        _force_reset_chroma_internals()
        get_collection()
        return _do()


def embed_image(image_path: str) -> list[float]:
    """Chinese-CLIP 把一张图片直接编码为 512D 向量。"""
    from PIL import Image

    model = _get_image_embedder()
    img = Image.open(image_path).convert("RGB")
    vec = model.encode(img, normalize_embeddings=True)
    return vec.tolist()


def add_image_to_kb(image_path: str, source_label: str = "") -> bool:
    """把单张图片嵌入到图片库；source_label 用于后续溯源。"""
    if os.getenv("ENABLE_IMAGE_RAG", "true").lower() == "false":
        return False
    try:
        coll = _get_image_collection()
        emb = embed_image(image_path)
        rid = f"{Path(image_path).name}::{hashlib.md5(open(image_path, 'rb').read()).hexdigest()[:8]}"
        meta = {"source": source_label or str(Path(image_path).name), "type": "image"}
        coll.upsert(ids=[rid], documents=[source_label or Path(image_path).name], metadatas=[meta], embeddings=[emb])
        return True
    except Exception as e:
        logger.warning("add_image_to_kb(%s) failed: %s", image_path, e)
        return False


def retrieve_images(query: str, top_k: int = 3) -> list[dict]:
    """用 Chinese-CLIP 文本编码查图片库；空库返回 []。"""
    if os.getenv("ENABLE_IMAGE_RAG", "true").lower() == "false":
        return []
    try:
        coll = _get_image_collection()
        if coll.count() == 0:
            return []
        model = _get_image_embedder()
        query_emb = model.encode(query, normalize_embeddings=True).tolist()
        result = coll.query(query_embeddings=[query_emb], n_results=top_k)
        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        dists = result.get("distances", [[]])[0]
        return [
            {
                "content": f"[图片] {doc}",
                "source": (meta or {}).get("source", "?"),
                "similarity": round(1.0 - float(dist), 4),
                "type": "image",
            }
            for doc, meta, dist in zip(docs, metas, dists)
        ]
    except Exception as e:
        logger.warning("retrieve_images failed: %s", e)
        return []


def hybrid_retrieve(query: str, top_k_text: int = 3, top_k_images: int = 2) -> list[dict]:
    """分别检索文本库（BGE）和图片库（Chinese-CLIP），按 similarity 合并。

    合并后排序：图片相似度需 >= IMAGE_SIMILARITY_MIN 才会被保留，
    避免低质量图片挤掉高相似度文本内容。
    """
    image_min = float(os.getenv("IMAGE_SIMILARITY_MIN", "0.35"))
    text_results = retrieve(query, top_k=top_k_text)
    image_results = retrieve_images(query, top_k=top_k_images)
    # 保留质量过线的图片结果
    image_results = [r for r in image_results if r.get("similarity", 0) >= image_min]
    merged = text_results + image_results
    merged.sort(key=lambda r: r.get("similarity", 0), reverse=True)
    return merged


def _reset_image_cache():
    """测试用。"""
    global _image_embedder
    _image_embedder = None
