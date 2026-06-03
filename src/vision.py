"""Vision Provider —— 把图片转成文本描述，让下游纯文本管道继续处理。

设计：
- 抽象 VisionProvider 接口
- 实际 provider: QwenVLProvider（阿里 DashScope OpenAI 兼容模式）
- 兜底 provider: MockVisionProvider（无 key 时返回提示）
- 自动选择：env 有 DASHSCOPE_API_KEY → Qwen-VL；否则 mock

集成点：brief_writer 收到 image_path 后调 describe_image()，把描述拼进 brief。
"""
from __future__ import annotations

import base64
import logging
import mimetypes
import os
from pathlib import Path
from typing import Protocol

logger = logging.getLogger(__name__)


# ---------- 协议 ----------

class VisionProvider(Protocol):
    """所有 vision provider 必须实现这个接口。"""

    name: str

    def describe_image(self, image_path: str, prompt: str = "") -> str:
        """把图片转成文本描述。失败时返回错误说明字符串，不抛异常。"""


# ---------- helpers ----------

def _encode_image_to_data_url(image_path: str) -> str:
    """读图 → base64 data URL（OpenAI / DashScope 兼容格式）。"""
    p = Path(image_path)
    if not p.exists():
        raise FileNotFoundError(f"图片不存在: {image_path}")
    mime, _ = mimetypes.guess_type(str(p))
    if not mime or not mime.startswith("image/"):
        mime = "image/png"  # 兜底
    b = p.read_bytes()
    b64 = base64.b64encode(b).decode("ascii")
    return f"data:{mime};base64,{b64}"


# ---------- Mock provider ----------

class MockVisionProvider:
    name = "mock"

    def describe_image(self, image_path: str, prompt: str = "") -> str:
        p = Path(image_path)
        exists = p.exists()
        size = p.stat().st_size if exists else 0
        return (
            f"[未配置 Vision LLM 后端（DASHSCOPE_API_KEY 未设置），返回 mock 描述]\n"
            f"图片路径: {image_path}\n"
            f"文件存在: {exists}, 大小: {size} bytes\n"
            f"prompt: {prompt[:200]}\n"
            f"提示：设置 DASHSCOPE_API_KEY 并选用 qwen-vl-max 等模型可启用真正的视觉理解。"
        )


# ---------- Qwen-VL provider（DashScope OpenAI 兼容模式） ----------

class QwenVLProvider:
    name = "qwen-vl-max"

    DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    DEFAULT_MODEL = "qwen-vl-max"

    def __init__(self, api_key: str | None = None, model: str | None = None, base_url: str | None = None):
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY", "")
        self.model = model or os.getenv("QWEN_VL_MODEL", self.DEFAULT_MODEL)
        self.base_url = base_url or os.getenv("DASHSCOPE_BASE_URL", self.DEFAULT_BASE_URL)

    def describe_image(self, image_path: str, prompt: str = "") -> str:
        if not self.api_key:
            return "[QwenVLProvider 未配置 DASHSCOPE_API_KEY]"
        try:
            from openai import OpenAI

            client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            data_url = _encode_image_to_data_url(image_path)
            user_text = prompt or "请详细描述这张图片的内容，包括关键文字、数据、人物、场景。"
            resp = client.chat.completions.create(
                model=self.model,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_url}},
                        {"type": "text", "text": user_text},
                    ],
                }],
            )
            return resp.choices[0].message.content or "(空响应)"
        except Exception as e:
            logger.warning("QwenVL describe_image failed: %s", e)
            return f"[Vision 调用异常: {type(e).__name__}: {e}]"


# ---------- 自动选择 ----------

def get_vision_provider() -> VisionProvider:
    """按 env 自动选 provider；DASHSCOPE_API_KEY 存在 → Qwen-VL；否则 mock。"""
    if os.getenv("DASHSCOPE_API_KEY", "").strip() and os.getenv("ENABLE_VISION", "true").lower() != "false":
        return QwenVLProvider()
    return MockVisionProvider()


def describe_image(image_path: str, prompt: str = "") -> str:
    """便捷入口。"""
    return get_vision_provider().describe_image(image_path, prompt)
