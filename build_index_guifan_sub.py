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
# ============================
class HybridChunker:
    def _split_by_article(self, text: str):
        pattern = r"(?=^\d{1,2}\.\d{1,2}\.\d{1,2}\s+)"
        parts = re.split(pattern, text, flags=re.MULTILINE)
        return [p.strip() for p in parts if len(p.strip()) > 0]

    def split(self, text: str):
        return [Document(page_content=c) for c in self._split_by_article(text)]


def clean_text(text):
    if not text:
        return ""
    text = text.replace("\x00", "")
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def load_pdf(path):
    text = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                text += clean_text(t) + "\n"
    return text

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
# 主流程：遍历 data 子目录
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

        # 遍历子目录下所有 PDF
        for file in os.listdir(sub_data):
            if file.endswith(".pdf"):
                pdf_path = os.path.join(sub_data, file)
                print(f"  ➜ 加载 PDF：{file}")
                text = load_pdf(pdf_path)
                docs.append(Document(page_content=text))

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
