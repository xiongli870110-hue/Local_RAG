# Local_RAG

本项目旨在构建一个“本地 RAG（Retrieval-Augmented Generation）”演示系统，基于本地文件（主要为 PDF）进行切片、向量化并构建本地向量数据库，然后用本地 LLM（通过 Ollama）结合 LangChain 做问答检索。

仓库说明
- 项目名称：Local_RAG
- 描述：对文件向量化，切片后构建本地知识库
- 语言：Python、Batchfile

核心功能
- 遍历 data/ 子目录，对每个子目录内的 PDF 文件进行文本提取与清洗。
- 使用自定义切片器（HybridChunker）按条文编号或章节规则切分文本片段。
- 使用 HuggingFaceEmbeddings 对切片进行向量化并用 FAISS 保存本地向量库（embeddings/）。
- 提供基于 Streamlit 的 Web UI（webui.py），支持从多个向量库并行检索（MMR）并调用本地 Ollama 模型生成回答。

文件结构（关键文件）
- build_index_guifan_sub.py  — 遍历 data/ 目录，对 PDF 提取文本并切片、生成向量库（保存到 embeddings/）。
- webui.py                 — Streamlit Web UI，加载 embeddings/ 下所有库并与本地 Ollama LLM 联动，提供聊天式 RAG 体验。
- start.bat                — Windows 下启动 Streamlit 的便捷脚本（通过虚拟环境的 python.exe）。
- data/                    — 存放待索引的子目录与 PDF（按书目分子目录）。
- embeddings/              — 生成的向量库目录（按拼音安全目录命名）。

快速开始

1. 环境要求
- 操作系统：Windows / macOS / Linux 均可（注意 start.bat 是 Windows 专用）。
- Python：推荐 3.10+。
- 建议使用虚拟环境（venv 或 conda）。
- 本地 Ollama（可选但推荐）：用于运行本地 LLM（webui.py 中使用 ChatOllama）。请先安装并配置 Ollama（参考 https://ollama.com/）。

2. 安装依赖（示例）
建议创建并激活虚拟环境后安装依赖。可以新建 requirements.txt 并包含下列依赖：

示例 requirements（可写入 requirements.txt 并执行 pip install -r requirements.txt）：

- streamlit
- pdfplumber
- faiss-cpu            # 或 faiss-gpu（如你有 GPU 环境并需要）
- pypinyin
- transformers
- huggingface-hub
- langchain-core
- langchain-huggingface
- langchain-ollama
- langchain-community

注意：部分 langchain 的扩展包命名或来源可能��随着时间变化，请根据你的环境和 pip 源调整包名和安装方法。

3. 准备数据
在仓库根目录下创建 data/ 目录，按书目创建子目录并把 PDF 放在对应子目录下。例如：

```
data/
  ├── 合同法（中文名）/
  │     ├── file1.pdf
  │     └── file2.pdf
  └── 民法典/
        └── another.pdf
```

脚本会把中文子目录名转换为拼音安全名，向量库最终保存在 embeddings/ 下对应的拼音目录中。

4. 构建向量索引

在激活的虚拟环境中运行：

```
python build_index_guifan_sub.py
```

脚本说明：
- 会遍历 data/ 下每个子目录，读取其中的 PDF 并用 pdfplumber 提取文本。
- 使用 HybridChunker 对文本进行切片（当前按条文编号的正则做分割，也可按需修改）。
- 使用 HuggingFaceEmbeddings（默认示例："BAAI/bge-large-zh"）对切片生成向量，并用 FAISS 保存到 embeddings/<拼音目录>/。

运行后可在 embeddings/ 下看到每个书目的向量库文件。

5. 启动 Web UI

方法一（Windows，使用仓库中提供的 start.bat）：

双击或在命令行调用：

```
start.bat
```

方法二（跨平台）：

```
python -m streamlit run webui.py
```

Web UI 功能要点：
- 自动加载 embeddings/ 下的所有向量库。
- 对用户输入问题，分别在每个库中做 max marginal relevance（MMR）检索，保证多源多样性。
- 用本地 Ollama 模型（webui.py 中默认 ChatOllama model="qwen2.5"）生成回答，且会显示对应的知识库片段作为引用来源。

配置项说明
- Embedding 模型：在 build_index_guifan_sub.py 中，embedder = HuggingFaceEmbeddings(model_name="BAAI/bge-large-zh"). 如需更换模型，请修改该行并确保相关模型可用。
- LLM（Ollama）：webui.py 中 llm = ChatOllama(model="qwen2.5", temperature=0.2)。请确认本地 Ollama 已安装目标模型（例如 qwen2.5）。
- FAISS 加载安全提示：webui.py 中加载本地 FAISS 时使用了 allow_dangerous_deserialization=True，若你遇到加载问题，检查 FAISS 与向量库版本是否一致。

设计细节与实现要点
- HybridChunker：自定义的切片器，当前以条文编号正则为主，位于 build_index_guifan_sub.py 的 HybridChunker 类。适合法规、条文类文档的分段。
- 中文目录转拼音：使用 pypinyin.lazy_pinyin 自动将子目录名转换为下划线分隔的拼音名，避免文件系统或库路径中的中文问题。
- MMR 检索策略：webui.py 中会从每个���量库检索若干候选（fetch_k），然后对每个库至少保留 1 条，再从剩余候选中补足总条数，保证多库覆盖性。

常见问题与排查
- 报错：找不到 Ollama 或 ChatOllama 抛错
  - 确认已安装并运行 Ollama 服务，并且系统中可以访问 Ollama 客户端。
  - 在 macOS/Linux 上需按 Ollama 官方文档安装并启动，并确保模型已下载（如 qwen2.5）。

- 向量库加载失败或反序列化报错
  - 确认 faiss 版本（faiss-cpu / faiss-gpu）与保存向量库时的版本一致。
  - webui.py 中使用 allow_dangerous_deserialization=True 是为了方便本地加载不同环境生成的索引，但若遇到兼容问题，建议重新用当前环境 build_index。

- 文本提取不完整或格式混乱
  - pdfplumber 对不同 PDF 的表现不同，可能需要针对复杂布局或扫描版 PDF 使用 OCR（例如 Tesseract）先转换为可检索文本。

安全与隐私
- 所有数据与模型均在本地运行与保存，不会上传到第三方服务（除非你的 embedding 或 LLM 配置指向云端服务）。
- 请注意嵌入模型或 LLM 可能会在联网时访问 HuggingFace 或其他服务，若需完全离线，请准备本地可用的模型及依赖并调整代码中模型路径。

扩展与自定义建议
- 切片策略：HybridChunker 当前以条文编号为准，可替换为基于长度、句子边界或语义的切片器（例如基于分段长度、TextSplitter 等）。
- Embedding：可替换为其它支持中文的 embedding 模型或本地 sentence-transformers 模型。
- 数据源：目前仅示例 PDF，可扩展为 Word、Markdown、HTML、数据库等来源，先行做文本提取后统一切片与向量化。
- 检索策略：可基于向量相似度 + 置信度阈值过滤，或结合稠密检索 + 布尔检索混合策略。

贡献
非常欢迎贡献改进：
1. Fork 本仓库并新建分支进行开发。
2. 提交 Pull Request 说明修改内容与测试方式。
3. 对重要改动（例如依赖、接口）请补充或更新 README 与示例。

许可证
本项目默认未指定许可证。建议根据需要添加许可证（例如 MIT）。

作者
- 仓库：xiongli870110-hue/Local_RAG


如果你希望我同时添加 requirements.txt、.gitignore 或者把 README 翻成英文版本，我可以继续按你的要求补充并提交。