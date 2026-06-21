# Local_RAG

本项目是一个本地化的 RAG（Retrieval-Augmented Generation）演示与工具集合，基于 Ollama + LangChain + 本地向量库（FAISS）。

主要功能：

- 扫描 data/ 子目录，支持 PDF 与 Markdown 文档，清洗后按规则切片（优先按 Markdown 标题切分，否则按条文编号切分），生成 chunk。
- 使用 HuggingFace Embeddings（示例：BAAI/bge-large-zh）为 chunk 生成向量，保存为本地 FAISS 向量库（embeddings/ 子目录）。
- 提供一个 Streamlit Web UI（webui.py），通过 Ollama 本地模型进行基于检索结果的回答。支持多向量库检索与公平抽取（MMR-like 策略）。
- 提供查看向量库内容的脚本（check_index_view_all.py），可以分页查看或导出为文本文件。

目录结构（关键文件）：

- data/  
  放置若干子目录（每个子目录代表一本书或一个知识集合），子目录中放 PDF (.pdf) 或 Markdown (.md/.markdown) 文件。
- build_index_guifan_md.py  
  构建向量索引的主脚本：加载 data 下每个子目录，读取 pdf/md，清洗并切片，使用 HuggingFaceEmbeddings 编码后保存为 FAISS 向量库到 embeddings/<子目录拼音>/。
- check_index_view_all.py  
  加载指定 embeddings 库并导出/分页显示全部 chunk，便于校验与人工查看（修改脚本顶部 EMB_DIR 可切换库）。
- webui.py  
  Streamlit Web 界面，加载 embeddings 下的所有子目录向量库，使用 Ollama 本地模型（langchain_ollama.ChatOllama）回答查询并展示命中 chunk 作为引用。
- start.bat  
  Windows 下用于启动 Streamlit 的脚本（通过 venv 中的 python 启动 streamlit run webui.py）。

使用说明

1) 环境准备

建议使用 Python 3.10+ 的虚拟环境（venv / conda）：

```bash
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

2) 安装依赖（示例）

请注意：某些包名或分发可能随时间变化，下列为根据项目代码给出的参考依赖包名：

```bash
pip install streamlit transformers pdfplumber pypinyin faiss-cpu
# 若分包名与下面不一致，请根据你的环境和 pip 仓库调整
pip install langchain-core langchain-huggingface langchain-community langchain-ollama
```

- 如果你使用 GPU、不同的 FAISS 版本或特殊平台，请按需安装对应的 faiss 包（如 faiss-gpu）。
- Ollama 相关：本项目通过本地 Ollama 服务与本地模型（示例：qwen2.5）通信，请先在本机安装并运行 Ollama（https://ollama.com/），并确保模型已加载或可用。

3) 准备文档数据

在仓库根目录创建 data/ 下的若干子目录，例如：

```
data/
  ├─ jie_gou_jian_ce/
  │    ├─ spec.md
  │    └─ annex.pdf
  └─ jie_gou_she_ji/
       └─ design.md
```

注意：脚本会自动将中文目录名转换为拼音作为 embeddings 的目录名（示例：结构检测 -> jiegou_jiance），以避免文件名/路径问题。

4) 构建向量索引

运行构建脚本，会在 embeddings/ 下为每个子目录生成向量库：

```bash
python build_index_guifan_md.py
```

过程说明：
- 支持 .pdf 和 .md/.markdown 文件。
- 文本清理：去 BOM、统一换行符、去除多余空白等。
- 切片规则（HybridChunker）：
  - 优先按 Markdown 标题（#、## 等）切分；
  - 若无 Markdown 标题，则按条文编号样式（如 1.1.1 开头）切分。
- 每个子目录会生成 embeddings/<pinyin_folder>/ 向量库目录。

5) 校验向量库（查看 chunk 内容）

使用 check_index_view_all.py：

```bash
python check_index_view_all.py
```

- 默认脚本顶部 EMB_DIR 指向某个向量库目录（示例：embeddings/jie_gou_she_ji），修改为你要查看的目录。
- 脚本会导出 embe.txt（包含所有 chunk 的文本）并允许分页查看（n/p/数字/q）。

6) 启动 Web UI

先确保 Ollama 服务可用且你已准备好一个可用的本地模型（webui.py 中默认 model="qwen2.5"，你可根据本地 Ollama 的模型名调整）。

本项目提供 Windows 启动脚本 start.bat（会使用仓库中 .venv 的 python 启动 streamlit）：

- Windows（使用 start.bat）
  双击 start.bat 或在命令行运行：

```powershell
.\start.bat
```

- 或者直接使用 python 启动 Streamlit：

```bash
python -m streamlit run webui.py
```

页面打开后输入问题，系统将：
- 对 embeddings/ 下每个向量库进行检索（每库采用 max_marginal_relevance_search），从每库至少抽取一条命中，然后汇总最多 3 条作为上下文。
- 将这些命中文档片段按编号送入 Ollama 的 LLM，生成回答，并在 UI 中展示来源片段。

重要配置点与提示

- Embedding 模型：
  - 代码默认使用 "BAAI/bge-large-zh"。根据实际环境，HuggingFace Embeddings 的���载方式可能需要相关 Hugging Face 权限或 token。
- Ollama：
  - 请确保 Ollama 已安装并且本地能用（https://ollama.com/）。webui.py 使用 langchain_ollama.ChatOllama 与本地 Ollama 服务通信。
- 向量库大小与搜索参数：
  - check_index_view_all.py 中 similarity_search k 参数设为 2000，用于导出全部 chunk；实际检索时 webui.py 对每库使用 fetch_k=20, k=3 等参数。
- 多库检索策略：
  - webui.py 对每个库至少保留一条结果，然后再从剩余结果补齐到最多 3 条（目的是保证多源公平性）。

常见问题与排查

- 无法加载 Embeddings/FAISS：
  - 确认模型能被 HuggingFaceEmbeddings 正确加载（网络/token/模型名）。
  - FAISS load_local 出错时，可能是序列化版本差异（allow_dangerous_deserialization=True 已设置，但不同环境的 FAISS/Python 版本仍可能不兼容）。
- Ollama 调用失败：
  - 确保 Ollama 服务正在运行，并且 webui.py 中的模型名与本地 Ollama 中一致。
- 构建后没有生成 embeddings：
  - 检查 data/ 下是否有可识别的 .pdf 或 .md 文件；检查 build_index_guifan_md.py 的输出日志。

贡献与许可

欢迎提交 issue 或 PR 来改进：例如：
- 支持其他 Embeddings 提供者（OpenAI、Cohere 等）。
- 改进切片策略（更智能的分段/合并策略）。
- 更完善的错误处理与日志。

如果你想要我把 README 进一步本地化（例如补充你常用的 Ollama/模型安装步骤、添加 CI 配置、或生成环境文件 requirements.txt），告诉我需要的细节，我可以继续完善并提交修改。
