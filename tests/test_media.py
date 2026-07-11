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


def _make_video(path, seconds=20, fps=2):
    import av
    from PIL import Image

    with av.open(str(path), "w") as container:
        stream = container.add_stream("mpeg4", rate=fps)
        stream.width, stream.height = 320, 240
        stream.pix_fmt = "yuv420p"
        for i in range(seconds * fps):
            img = Image.new("RGB", (320, 240), (min(30 + i, 255), 60, 120))
            for packet in stream.encode(av.VideoFrame.from_image(img)):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def test_extract_keyframes_interval(tmp_path):
    from agentic_rag.media import extract_keyframes

    video = tmp_path / "clip.mp4"
    _make_video(video, seconds=20, fps=2)
    frames = extract_keyframes(str(video), every_seconds=10)
    assert len(frames) == 2
    assert abs(frames[0][0] - 0.0) < 1.0
    assert abs(frames[1][0] - 10.0) < 1.0
    assert frames[0][1][:8] == b"\x89PNG\r\n\x1a\n", "帧应编码为 PNG"


def test_video_to_documents_merges_audio_and_frames(tmp_path):
    from agentic_rag.media import video_to_documents

    video = tmp_path / "review.mp4"
    _make_video(video, seconds=20, fps=2)

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
