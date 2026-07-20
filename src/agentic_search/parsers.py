"""文档解析层:把各种格式归一化为 markdown 文本(文本是索引,原始文件是档案)。"""
from pathlib import Path

SUPPORTED_EXTENSIONS = {".md", ".pdf"}


class UnsupportedFormatError(ValueError):
    """不支持的文件格式。"""


def parse_file(path: Path) -> str:
    """把单个文件解析为 markdown 文本;不支持的扩展名抛 UnsupportedFormatError。"""
    suffix = path.suffix.lower()
    if suffix == ".md":
        return path.read_text(encoding="utf-8")
    if suffix == ".pdf":
        import pymupdf4llm

        return pymupdf4llm.to_markdown(str(path))
    raise UnsupportedFormatError(f"不支持的格式: {suffix} ({path.name})")
