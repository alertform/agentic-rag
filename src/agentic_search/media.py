"""媒体归一化管道:音频 ASR 转写、视频关键帧 VLM 描述 → 带时间戳 locator 的文本块。

设计:媒体的"位置"是时间轴而非标题路径,复用 metadata 的 headers 字段作 locator
(如 "00:01:00 - 00:02:00"),来源标注与引用汇总零改动直接生效。
文本是索引,原始媒体是档案:locator 指回原始文件的具体时刻。
"""
from typing import Callable, NamedTuple

from langchain_core.documents import Document

SUPPORTED_AUDIO = {".wav", ".mp3", ".m4a"}
SUPPORTED_VIDEO = {".mp4", ".mov"}


class Segment(NamedTuple):
    start: float
    end: float
    text: str


Transcriber = Callable[[str], list[Segment]]
Captioner = Callable[[bytes], str]


def _format_ts(seconds: float) -> str:
    total = int(seconds)
    return f"{total // 3600:02d}:{total % 3600 // 60:02d}:{total % 60:02d}"


def segments_to_documents(
    segments: list[Segment], source: str, window_seconds: int = 60
) -> list[Document]:
    """按时间窗聚合转写分段;跨窗长段归属其起始窗口。"""
    windows: dict[int, list[str]] = {}
    for seg in segments:
        if seg.text.strip():
            windows.setdefault(int(seg.start // window_seconds), []).append(seg.text.strip())
    docs = []
    for idx in sorted(windows):
        locator = (
            f"{_format_ts(idx * window_seconds)} - {_format_ts((idx + 1) * window_seconds)}"
        )
        docs.append(
            Document(
                page_content=" ".join(windows[idx]),
                metadata={"source": source, "headers": locator},
            )
        )
    return docs


def extract_keyframes(path: str, every_seconds: int = 10) -> list[tuple[float, bytes]]:
    """按固定时间间隔抽取视频帧,返回 (时间戳秒, PNG bytes) 列表。"""
    import io

    import av

    frames: list[tuple[float, bytes]] = []
    with av.open(path) as container:
        stream = container.streams.video[0]
        next_ts = 0.0
        for frame in container.decode(stream):
            ts = float(frame.time or 0.0)
            if ts + 1e-6 >= next_ts:
                buf = io.BytesIO()
                frame.to_image().save(buf, format="PNG")
                frames.append((ts, buf.getvalue()))
                next_ts += every_seconds
    return frames


def make_ollama_captioner() -> Captioner:
    """VLM 帧描述:走本地 Ollama 的 vision 能力(生成模型即 qwen3.5,零额外模型)。"""
    import base64
    import json
    import urllib.request

    from agentic_search import config

    def caption(image_png: bytes) -> str:
        payload = {
            "model": config.GENERATION_MODEL,
            "stream": False,
            "messages": [
                {
                    "role": "user",
                    "content": "用一两句话描述这帧画面;若画面含文字,请原样转录全部文字。",
                    "images": [base64.b64encode(image_png).decode()],
                }
            ],
        }
        req = urllib.request.Request(
            f"{config.OLLAMA_BASE_URL}/api/chat",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=300) as resp:
            data = json.load(resp)
        return data["message"]["content"].strip()

    return caption


def video_to_documents(
    path: str,
    source: str,
    transcriber: Transcriber,
    captioner: Captioner,
    window_seconds: int = 60,
    frame_interval: int = 10,
) -> list[Document]:
    """视频归一化:音轨转写 + 关键帧描述,统一带时间轴 locator。

    音轨转写失败直接向上抛(由 ingest 侧决定跳过该文件而非缓存残缺结果)——
    避免"只剩画面"的半吊子索引被媒体缓存永久固化、不可自愈。
    """
    docs = segments_to_documents(transcriber(str(path)), source, window_seconds)
    for ts, png in extract_keyframes(str(path), frame_interval):
        text = captioner(png).strip()
        if text:
            docs.append(
                Document(
                    page_content=text,
                    metadata={"source": source, "headers": f"画面 {_format_ts(ts)}"},
                )
            )
    return docs


def cached_media_documents(file_path, source: str, cache_dir, builder) -> list[Document]:
    """媒体解析缓存:按文件内容 sha256 缓存 builder() 的产物。

    ASR/VLM 是采样式解析,同一文件重复解析文本会微变 → 块哈希变 → 增量索引被
    无意义搅动,且白付模型开销。文件不变即复用缓存,变了才重新解析。
    """
    import hashlib
    import json
    from pathlib import Path

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(Path(file_path).read_bytes()).hexdigest()[:32]
    cache_file = cache_dir / f"{digest}.json"
    if cache_file.is_file():
        payload = json.loads(cache_file.read_text(encoding="utf-8"))
        return [Document(page_content=e["page_content"], metadata=e["metadata"]) for e in payload]

    docs = builder()
    cache_file.write_text(
        json.dumps(
            [{"page_content": d.page_content, "metadata": d.metadata} for d in docs],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return docs


def make_whisper_transcriber(model_size: str = "small") -> Transcriber:
    """懒加载 faster-whisper。

    优先用本地模型目录 models/faster-whisper-<size>(可 curl 从镜像直接下载,
    见 README);否则经 HF_ENDPOINT 镜像下载(默认 hf-mirror)。
    """
    import os

    from agentic_search import config

    local_dir = config.PROJECT_ROOT / "models" / f"faster-whisper-{model_size}"
    target = str(local_dir) if local_dir.is_dir() else model_size
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    from faster_whisper import WhisperModel

    # CPU int8:small 模型转写快于实时,且不依赖 CUDA 运行库(cublas/cudnn)
    model = WhisperModel(target, device="cpu", compute_type="int8")

    def transcribe(path: str) -> list[Segment]:
        raw_segments, _info = model.transcribe(path, language="zh")
        return [Segment(s.start, s.end, s.text) for s in raw_segments]

    return transcribe
