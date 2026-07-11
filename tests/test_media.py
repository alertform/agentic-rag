"""媒体管道测试:时间窗分段 / 关键帧抽取 / 视频块结构。零模型依赖。"""
from agentic_rag.media import Segment, segments_to_documents


def test_segments_group_into_windows():
    segs = [
        Segment(0.0, 5.0, "第一句。"),
        Segment(58.0, 62.0, "跨窗句归属起始窗口。"),
        Segment(65.0, 70.0, "第二窗的句子。"),
    ]
    docs = segments_to_documents(segs, "meeting.wav", window_seconds=60)
    assert len(docs) == 2
    assert docs[0].metadata["source"] == "meeting.wav"
    assert docs[0].metadata["headers"] == "00:00:00 - 00:01:00"
    assert "第一句。" in docs[0].page_content
    assert "跨窗句归属起始窗口。" in docs[0].page_content
    assert docs[1].metadata["headers"] == "00:01:00 - 00:02:00"
    assert "第二窗的句子。" in docs[1].page_content


def test_empty_segments_returns_empty():
    assert segments_to_documents([], "a.wav") == []


def test_locator_format_with_hours():
    segs = [Segment(3661.0, 3670.0, "一小时后的内容。")]
    docs = segments_to_documents(segs, "long.mp3", window_seconds=60)
    assert docs[0].metadata["headers"] == "01:01:00 - 01:02:00"


def test_extract_keyframes_interval(tmp_path):
    from conftest import make_video

    from agentic_rag.media import extract_keyframes

    video = tmp_path / "clip.mp4"
    make_video(video, seconds=20, fps=2)
    frames = extract_keyframes(str(video), every_seconds=10)
    assert len(frames) == 2
    assert abs(frames[0][0] - 0.0) < 1.0
    assert abs(frames[1][0] - 10.0) < 1.0
    assert frames[0][1][:8] == b"\x89PNG\r\n\x1a\n", "帧应编码为 PNG"


def test_load_documents_routes_media(tmp_path):
    from conftest import make_video

    from agentic_rag.ingest import load_documents

    (tmp_path / "menu.md").write_text("# 菜单\n\n拿铁 32 元。", encoding="utf-8")
    (tmp_path / "meeting.wav").write_bytes(b"fake-bytes")  # 假转写器不读内容
    make_video(tmp_path / "review.mp4", seconds=10, fps=2)

    def fake_transcriber(path):
        return [Segment(0.0, 3.0, "会议决议内容。")]

    docs = load_documents(
        tmp_path, transcriber=fake_transcriber, captioner=lambda png: "幻灯片画面"
    )
    sources = {d.metadata["source"] for d in docs}
    assert sources == {"menu.md", "meeting.wav", "review.mp4"}
    assert all(d.metadata["access"] == "public" for d in docs), "媒体块也应打 access 标"


def test_media_parse_cache_avoids_reprocessing(tmp_path):
    from agentic_rag.ingest import load_documents

    (tmp_path / "meeting.wav").write_bytes(b"audio-bytes-v1")
    calls = {"n": 0}

    def counting_transcriber(path):
        calls["n"] += 1
        return [Segment(0.0, 3.0, "会议内容。")]

    cache_dir = tmp_path / ".cache"
    docs1 = load_documents(tmp_path, transcriber=counting_transcriber, media_cache_dir=cache_dir)
    docs2 = load_documents(tmp_path, transcriber=counting_transcriber, media_cache_dir=cache_dir)
    assert calls["n"] == 1, "文件未变,第二次应命中解析缓存"
    assert [d.page_content for d in docs1] == [d.page_content for d in docs2]
    assert docs2[0].metadata["headers"] == "00:00:00 - 00:01:00"

    # 文件内容变了 → 缓存失效,重新转写
    (tmp_path / "meeting.wav").write_bytes(b"audio-bytes-v2")
    load_documents(tmp_path, transcriber=counting_transcriber, media_cache_dir=cache_dir)
    assert calls["n"] == 2


def test_media_skipped_without_transcriber(tmp_path):
    from agentic_rag.ingest import load_documents

    (tmp_path / "a.md").write_text("# A\n\n内容。", encoding="utf-8")
    (tmp_path / "meeting.wav").write_bytes(b"fake-bytes")
    docs = load_documents(tmp_path)
    assert {d.metadata["source"] for d in docs} == {"a.md"}


def test_video_to_documents_merges_audio_and_frames(tmp_path):
    from conftest import make_video

    from agentic_rag.media import video_to_documents

    video = tmp_path / "review.mp4"
    make_video(video, seconds=20, fps=2)

    def fake_transcriber(path):
        return [Segment(0.0, 5.0, "这是评审会的开场。")]

    def fake_captioner(png):
        return "幻灯片文字: 新品评审 SP-2026"

    docs = video_to_documents(
        str(video), "review.mp4", fake_transcriber, fake_captioner, frame_interval=10
    )
    audio_docs = [d for d in docs if "开场" in d.page_content]
    frame_docs = [d for d in docs if "SP-2026" in d.page_content]
    assert len(audio_docs) == 1
    assert audio_docs[0].metadata["headers"] == "00:00:00 - 00:01:00"
    assert len(frame_docs) == 2
    assert frame_docs[0].metadata["headers"] == "画面 00:00:00"
    assert all(d.metadata["source"] == "review.mp4" for d in docs)
