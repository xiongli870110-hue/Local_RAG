import os
import re
import shutil
import pdfplumber
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from pypinyin import lazy_pinyin

# ============================
# 条文编号切片器（你已经验证成功）
# 现在增加了对 Markdown 标题的支持：优先按标题切分，否则按条文编号切分
# ============================
class HybridChunker:
    def _split_by_article(self, text: str):
        pattern = r"(?=^\d{1,2}\.\d{1,2}\.\d{1,2}\s+)"
        parts = re.split(pattern, text, flags=re.MULTILINE)
        return [p.strip() for p in parts if len(p.strip()) > 0]

    def _split_by_markdown_header(self, text: str):
        # 使用对每个 markdown 标题的前瞻切分（保留标题与其内容）
        pattern = r"(?=^#{1,6}\s+)"
        parts = re.split(pattern, text, flags=re.MULTILINE)
        return [p.strip() for p in parts if len(p.strip()) > 0]

    def split(self, text: str):
        if not text:
            return []
        # 如果文本包含 Markdown 标题，优先按标题切分
        if re.search(r"(?m)^#{1,6}\s+", text):
            parts = self._split_by_markdown_header(text)
        else:
            parts = self._split_by_article(text)
        return [Document(page_content=c) for c in parts]


def clean_text(text):
    if not text:
        return ""
    # 删除空字符并把连续空白化为单个空格
    text = text.replace("\x00", "")
    # 去掉 BOM
    text = text.lstrip("\ufeff")
    text = re.sub(r"[ \t]+", " ", text)
    # 统一换行符（保留换行以便按段落/标题拆分）
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text.strip()


def load_pdf(path):
    text = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                text += clean_text(t) + "\n"
    return text

def load_md(path):
    """
    读取 markdown 文件为纯文本。简单清理 BOM 与多余空白。
    如果需要去掉 front-matter (YAML) 或代码块，可在这里扩展。
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()
    # 去掉常见的 YAML front-matter（--- 开头结尾）
    raw = re.sub(r"(?s)^---.*?---\n", "", raw)
    return clean_text(raw)

# ============================
# 中文目录自动转换为拼音目录
# ============================
def chinese_to_pinyin(name):
  """
  自动将中文目录名转换为拼音目录名
  结构检测 → jiegou_jiance
  """
  # 转拼音
  pinyin_list = lazy_pinyin(name)
  pinyin = "_".join(pinyin_list)

  # 清理非法字符
  pinyin = re.sub(r"[^a-zA-Z0-9_]", "_", pinyin)

  return pinyin

# ============================
# 主流程：遍历 data 子目录，支持 .pdf, .md, .markdown
# ============================
def build_all_indexes():
    DATA_DIR = "data"
    EMB_DIR = "embeddings"

    embedder = HuggingFaceEmbeddings(model_name="BAAI/bge-large-zh")
    chunker = HybridChunker()

    for folder in os.listdir(DATA_DIR):
        sub_data = os.path.join(DATA_DIR, folder)
        if not os.path.isdir(sub_data):
            continue

        print(f"\n📘 处理书目：{folder}")

        # 输出目录
        safe_folder = chinese_to_pinyin(folder)
        sub_emb = os.path.join(EMB_DIR, safe_folder)
        if os.path.exists(sub_emb):
            shutil.rmtree(sub_emb)
        os.makedirs(sub_emb, exist_ok=True)

        docs = []

        # 遍历子目录下所有文件，支持 pdf / md / markdown
        for file in os.listdir(sub_data):
            lower = file.lower()
            file_path = os.path.join(sub_data, file)
            if lower.endswith(".pdf"):
                print(f"  ➜ 加载 PDF：{file}")
                text = load_pdf(file_path)
                if text:
                    docs.append(Document(page_content=text, metadata={"source": file}))
            elif lower.endswith(".md") or lower.endswith(".markdown"):
                print(f"  ➜ 加载 MD：{file}")
                text = load_md(file_path)
                if text:
                    docs.append(Document(page_content=text, metadata={"source": file}))

        if not docs:
            print("  ⚠️ 未发现可处理的文件（pdf/md）")
            continue

        # 切片
        chunks = []
        for d in docs:
            chunks.extend(chunker.split(d.page_content))

        print(f"  ✂️ chunk 数量：{len(chunks)}")

        # 构建向量库
        db = FAISS.from_documents(chunks, embedder)
        db.save_local(sub_emb)

        print(f"  🎉 已生成向量库：{sub_emb}")


if __name__ == "__main__":
    build_all_indexes()
