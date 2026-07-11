# 企业级演进 Phase 1:多格式文档 + 增量索引 + 混合检索

**Goal:** 把 agentic-rag 从"只吃 Markdown 的 demo"推向企业数据形态:PDF 等文档归一化接入、增量索引替代全量重建、BM25+向量混合检索。

**背景**(设计讨论结论):企业 RAG 的核心模式是"一切数据归一化为『带丰富 metadata 的文本块』,文本是索引、不是档案,metadata 永远指回原始资产"。文档类数据的最佳中间格式是 markdown(标题层级直接对接现有 `MarkdownHeaderTextSplitter`)。音视频(ASR/VLM 转写)属 Phase 2——受本机模型下载网络约束,另开计划。

## Global Constraints

- 沿用现有约束:Python 3.12 + uv、全本地、单测零 Ollama 依赖
- 解析器选型 **pymupdf4llm**(PDF→markdown,纯 pip 无模型下载,适配本机网络;MinerU/Docling 需下载模型,列为可替换项)
- 中文 BM25 分词用 **jieba**(纯 Python)
- 每 Task 结束 commit,中文 commit message

## Task 1: 文档解析层(TDD)

**Files:** Create `src/agentic_rag/parsers.py`;Test `tests/test_parsers.py`;Modify `ingest.py` 的 `load_documents`

**Produces:** `parse_file(path: Path) -> str`(返回 markdown 文本;`.md` 直读,`.pdf` 走 pymupdf4llm,未知扩展名抛 `UnsupportedFormatError`);`SUPPORTED_EXTENSIONS: set[str]`;`load_documents` 改为按 `SUPPORTED_EXTENSIONS` 递归收集并调 `parse_file`,source 仍为相对路径(保留原始扩展名)

**Tests:** md 直读内容一致;用 pymupdf 程序化生成单页 PDF(tmp_path)→ parse 出的文本含预期字符串;未知格式抛错;load_documents 混合目录(md+pdf+txt)只收支持的格式

## Task 2: 增量索引(TDD)

**Files:** Modify `src/agentic_rag/ingest.py`;Test `tests/test_ingest.py` 追加

**Produces:** `chunk_id(doc: Document) -> str`(source+headers+content 的 sha256 前 32 位,稳定幂等);`build_vector_store(chunks, rebuild: bool = False)`:默认增量——`add_documents(ids=...)` upsert 新增/变更块,再删除"库里有但本次未出现"的 id(按 source 维度对账);`--rebuild` CLI 参数走原 `reset_collection` 全量路径

**Tests:** 用 `DeterministicFakeEmbedding`(langchain-core)+ 临时目录 Chroma:两次 ingest 相同语料 → 块数不变、id 稳定;改一个文件再 ingest → 旧块消失新块在;删一个文件再 ingest → 对应块被清除。零 Ollama。

## Task 3: 混合检索 BM25+向量 RRF(TDD)

**Files:** Create `src/agentic_rag/retrieval.py`;Test `tests/test_retrieval.py`;Modify `chat.py` 接线

**Produces:** `HybridRetriever(vector_store, docs: list[Document], k_each: int = 20)`,暴露 `similarity_search(query, k)`(鸭子类型兼容 `make_retrieve_tool`):jieba 分词 BM25(rank_bm25)与向量检索各取 top-k_each,RRF(k=60)融合取 top-k;`load_all_chunks(store) -> list[Document]`(从 Chroma 取全量块供 BM25 建索引);chat.py 用 HybridRetriever 包装 store 后再 `make_retrieve_tool`

**Tests:** stub 向量库(可控返回)+ 小中文语料:纯词面命中(如编号/专名"NX-42")BM25 能捞回而向量 stub 不返回 → 融合结果含它;RRF 排序符合公式;k 截断正确。零 Ollama。

## Task 4: 语料扩充 + 手动验收 + README

- `sample_docs/` 增加一份 PDF(用脚本从"星尘咖啡馆供应商价目表"类虚构内容生成,含编号类专名,便于验证 BM25 价值)
- 手动验收(Ollama 机):`ingest` 混合语料 → chat 问 PDF 里的编号类问题,验证检索命中 PDF 来源并正确引用;重跑 ingest 验证增量(输出"新增 0 / 删除 0")
- README:格式支持矩阵、增量说明、混合检索说明、企业演进路线(Phase 2:音视频 ASR/VLM、权限过滤、重排序)

## 完成定义

- 全部单测通过且零 Ollama 依赖;手动验收三项通过;README 更新;逐 task 已提交并推送
