"""共享测试工具:合成媒体文件。"""
import av
from PIL import Image


def make_video(path, seconds=20, fps=2, text=None):
    """PyAV+Pillow 合成测试视频;text 非 None 时把文字画在每帧上(需系统中文字体)。"""
    font = None
    if text is not None:
        from PIL import ImageDraw, ImageFont

        font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 28)
    with av.open(str(path), "w") as container:
        stream = container.add_stream("mpeg4", rate=fps)
        stream.width, stream.height = 640, 360
        stream.pix_fmt = "yuv420p"
        for i in range(seconds * fps):
            img = Image.new("RGB", (640, 360), (245, 245, 240))
            if text is not None:
                from PIL import ImageDraw

                ImageDraw.Draw(img).text((40, 150), text, fill=(20, 20, 20), font=font)
            for packet in stream.encode(av.VideoFrame.from_image(img)):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
