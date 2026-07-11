# 企业级演进 Phase 2:音视频接入 + 权限过滤(ACL)

**Goal:** 音频/视频归一化为"带时间戳 locator 的文本块"进入现有管道;检索按提问者角色过滤可见范围。

**核心设计**(承 Phase 1 的归一化原则):
- 媒体的"位置"不是标题路径而是时间轴 → **复用 `headers` 字段作 locator**(如 `00:01:00 - 00:02:00`),`format_chunks` 的来源标注 `[来源: meeting.wav | 00:01:00 - 00:02:00]` 零改动直接生效
- 音频:ASR 转写(faster-whisper,HF_ENDPOINT 镜像下载)→ 按时间窗聚合分段
- 视频:音轨走 ASR;画面按间隔抽关键帧 → VLM(Ollama qwen3.5:9b vision)描述/OCR → 与转写同轴合并
- ACL:语料目录放 `acl.json`(glob → access 级别),块 metadata 带 `access`;检索时按角色可见级别过滤(向量走 Chroma where,BM25 预过滤)

## Global Constraints

- 单测**零模型依赖**:ASR/VLM 以可调用对象注入,测试用脚本化假实现;PyAV 合成视频可直接测关键帧抽取
- 真实 ASR/VLM 只出现在 CLI 主路径与手动验收
- 新依赖:faster-whisper、av、pillow(合成测试媒体)
- 重排序(reranker)不在本计划:需跨编码器模型,列为 Phase 3

## Task 1: 媒体转写与时间窗分段(TDD)

**Files:** Create `src/agentic_rag/media.py`;Test `tests/test_media.py`

**Produces:** `Segment`(start/end/text 三元组);`segments_to_documents(segments, source, window_seconds=60) -> list[Document]`(按时间窗聚合,headers 为 `HH:MM:SS - HH:MM:SS` locator);`make_whisper_transcriber(model_size="small")`(懒加载 faster-whisper,返回 `path -> list[Segment]` 可调用);`SUPPORTED_AUDIO`、`SUPPORTED_VIDEO`

**Tests:** 分段聚合窗口边界正确;locator 格式正确;空转写返回空列表;跨窗口长段归属起始窗口

## Task 2: 视频关键帧 + VLM 描述(TDD)

**Files:** Modify `src/agentic_rag/media.py`;Test `tests/test_media.py` 追加

**Produces:** `extract_keyframes(path, every_seconds=10) -> list[tuple[float, bytes]]`(PyAV 解码按间隔取帧,PNG bytes);`make_ollama_captioner()`(Ollama /api/chat 带 images 调 qwen3.5 描述画面,懒加载);`video_to_documents(path, source, transcriber, captioner, window_seconds=60)`(音轨转写 + 帧描述合并,帧描述块 headers 为 `画面 HH:MM:SS`)

**Tests:** 用 PyAV+Pillow 合成 20 秒纯色带文字视频 → 抽帧数量/时间戳正确;假 captioner/假 transcriber 下 video_to_documents 的块结构与 locator 正确

## Task 3: ACL 权限过滤(TDD)

**Files:** Create `src/agentic_rag/acl.py`;Modify `ingest.py`(块打 access 标)、`retrieval.py`(过滤)、`chat.py`(--role);Test `tests/test_acl.py`

**Produces:** `load_acl(docs_dir) -> list[(pattern, access)]`(读 `acl.json`,无文件全 public);`access_for(source, rules) -> str`(首个匹配 glob 生效,默认 public);`ROLE_ACCESS: dict[str, set[str]]`(config:public→{public}, staff→{public,internal}, manager→全部);`HybridRetriever(..., allowed_access: set[str] | None)`:向量走 Chroma `filter={"access": {"$in": [...]}}`,BM25 建索引前预过滤;chat `--role staff` 接线

**Tests:** glob 匹配优先级;无 acl.json 全 public;检索过滤:internal 块对 public 角色不可见、对 staff 可见(假向量库 + 真 BM25);ingest 块 metadata 带 access

## Task 4: ingest 路由媒体 + 语料生成 + 手动验收 + README

- `load_documents` 扩展:audio/video 扩展名路由到媒体管道(transcriber/captioner 参数化,CLI 侧仅在存在媒体文件时构造真实实现)
- 语料:Windows SAPI TTS 合成"星尘咖啡馆例会录音"(念虚构决议,如"下月起彗星气泡美式调价到 28 元")wav;PyAV+Pillow 合成含文字幻灯片("新品评审 SP-2026")的短视频;`acl.json` 把 suppliers.pdf 标 internal
- 手动验收(Ollama 机):① chat 问例会决议 → 命中 wav 来源并带时间戳引用;② 问 SP-2026 → 命中视频帧描述;③ `--role public` 问 NX-42 → 检索不到(ACL 生效),`--role staff` 能答
- README:媒体格式矩阵、ACL 用法、Phase 3 路线(重排序、多租户)

## 完成定义

全部单测通过且零模型依赖;三项手动验收通过;README 更新;逐 task 提交推送
