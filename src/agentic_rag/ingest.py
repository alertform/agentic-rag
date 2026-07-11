"""离线索引管道:文档目录 → 解析归一化 → 切块 → 向量化 → Chroma(增量同步)。"""
import hashlib
import sys
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from agentic_rag import config

_HEADERS_TO_SPLIT_ON = [("#", "h1"), ("##", "h2"), ("###", "h3")]


def split_markdown(text: str, source: str) -> list[Document]:
    """按标题层级切块,超长小节按字符二次切分;metadata 带 source 和标题路径。"""
    header_splitter = MarkdownHeaderTextSplitter(_HEADERS_TO_SPLIT_ON, strip_headers=False)
    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP
    )
    chunks: list[Document] = []
    for section in header_splitter.split_text(text):
        headers = " > ".join(
            section.metadata[key]
            for key in ("h1", "h2", "h3")
            if section.metadata.get(key)
        )
        for piece in char_splitter.split_documents([section]):
            piece.metadata = {"source": source, "headers": headers}
            chunks.append(piece)
    return chunks


def load_documents(
    docs_dir: Path, transcriber=None, captioner=None, media_cache_dir=None
) -> list[Document]:
    """递归加载目录下所有受支持格式的文档/媒体,归一化为带 metadata 的文本块。

    - 文档(md/pdf):解析为 markdown 后按标题切块
    - 音频:需注入 transcriber,按时间窗分段;视频:另需 captioner 做帧描述
    - 媒体解析结果按文件内容哈希缓存(media_cache_dir),文件不变不重解析
    - 未注入转写/描述器时跳过对应媒体文件
    - 块 metadata 按目录 acl.json 打 access 标(无规则默认 public)
    """
    from agentic_rag.acl import access_for, load_acl
    from agentic_rag.media import (
        SUPPORTED_AUDIO,
        SUPPORTED_VIDEO,
        cached_media_documents,
        segments_to_documents,
        video_to_documents,
    )
    from agentic_rag.parsers import SUPPORTED_EXTENSIONS, parse_file

    if media_cache_dir is None:
        media_cache_dir = config.MEDIA_CACHE_DIR
    rules = load_acl(docs_dir)
    chunks: list[Document] = []
    for file in sorted(p for p in docs_dir.rglob("*") if p.is_file()):
        suffix = file.suffix.lower()
        source = file.relative_to(docs_dir).as_posix()
        if suffix in SUPPORTED_EXTENSIONS:
            chunks.extend(split_markdown(parse_file(file), source))
        elif suffix in SUPPORTED_AUDIO and transcriber is not None:
            chunks.extend(
                cached_media_documents(
                    file, source, media_cache_dir,
                    lambda f=file, s=source: segments_to_documents(transcriber(str(f)), s),
                )
            )
        elif suffix in SUPPORTED_VIDEO and transcriber is not None and captioner is not None:
            chunks.extend(
                cached_media_documents(
                    file, source, media_cache_dir,
                    lambda f=file, s=source: video_to_documents(str(f), s, transcriber, captioner),
                )
            )
    for chunk in chunks:
        chunk.metadata["access"] = access_for(chunk.metadata["source"], rules)
    return chunks


def chunk_id(doc: Document) -> str:
    """块的稳定 ID:source+headers+access+内容 的 sha256 前 32 位。

    access 参与哈希:ACL 规则变更时旧块被替换,metadata 才能跟着更新。
    """
    meta = doc.metadata
    key = (
        f"{meta['source']}|{meta['headers']}|"
        f"{meta.get('access', 'public')}|{doc.page_content}"
    )
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:32]


def sync_vector_store(store, chunks: list[Document]) -> tuple[int, int]:
    """增量对账:只嵌入新增/变更的块,删除库里已消失的块;返回 (新增数, 删除数)。"""
    desired = {chunk_id(c): c for c in chunks}
    existing = set(store.get()["ids"])
    new_ids = [i for i in desired if i not in existing]
    stale_ids = [i for i in existing if i not in desired]
    if new_ids:
        store.add_documents([desired[i] for i in new_ids], ids=new_ids)
    if stale_ids:
        store.delete(ids=stale_ids)
    return len(new_ids), len(stale_ids)


def build_vector_store(chunks: list[Document], rebuild: bool = False):
    """打开本地 Chroma 并同步块:默认增量;rebuild=True 时清空全量重建。"""
    from langchain_chroma import Chroma
    from langchain_ollama import OllamaEmbeddings

    embeddings = OllamaEmbeddings(
        model=config.EMBEDDING_MODEL, base_url=config.OLLAMA_BASE_URL
    )
    store = Chroma(
        collection_name=config.COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(config.CHROMA_DIR),
    )
    if rebuild:
        store.reset_collection()
    added, removed = sync_vector_store(store, chunks)
    return store, added, removed


def main() -> None:
    from agentic_rag.media import SUPPORTED_AUDIO, SUPPORTED_VIDEO
    from agentic_rag.preflight import check_ollama

    args = sys.argv[1:]
    rebuild = "--rebuild" in args
    positional = [a for a in args if not a.startswith("--")]
    docs_dir = Path(positional[0]) if positional else config.SAMPLE_DOCS_DIR
    if not docs_dir.is_dir():
        sys.exit(f"[ingest] 目录不存在: {docs_dir}")

    suffixes = {p.suffix.lower() for p in docs_dir.rglob("*") if p.is_file()}
    has_audio = bool(suffixes & SUPPORTED_AUDIO)
    has_video = bool(suffixes & SUPPORTED_VIDEO)
    # 视频帧描述走生成模型的 vision 能力,故有视频时也要求生成模型就绪
    check_ollama(require_generation=has_video)

    transcriber = captioner = None
    if has_audio or has_video:
        from agentic_rag.media import make_whisper_transcriber

        print("[ingest] 检测到媒体文件,加载 ASR 模型…", flush=True)
        transcriber = make_whisper_transcriber()
    if has_video:
        from agentic_rag.media import make_ollama_captioner

        captioner = make_ollama_captioner()

    chunks = load_documents(docs_dir, transcriber=transcriber, captioner=captioner)
    if not chunks:
        sys.exit(f"[ingest] {docs_dir} 下没有受支持格式的文档 (md/pdf/wav/mp3/mp4)")
    store, added, removed = build_vector_store(chunks, rebuild=rebuild)
    mode = "全量重建" if rebuild else "增量同步"
    print(
        f"[ingest] {mode}完成: 共 {len(chunks)} 块 (新增 {added}, 删除 {removed}) ← {docs_dir}"
    )

    # 语料一变就跑一次检索评估(仅默认语料目录且 golden set 存在;失败不阻塞 ingest)
    golden_path = config.PROJECT_ROOT / "sample_evals.jsonl"
    if docs_dir.resolve() == config.SAMPLE_DOCS_DIR.resolve() and golden_path.is_file():
        from agentic_rag.evals import run_evals

        try:
            run_evals(store, chunks, golden_path)
        except Exception as exc:  # 评估是观测手段,不应让索引失败
            print(f"[ingest] 评估跳过: {exc}")


if __name__ == "__main__":
    main()
