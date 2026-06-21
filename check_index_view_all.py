import math
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


#EMB_DIR = "embeddings/jie_gou_jian_ce"   # 向量库目录
EMB_DIR = "embeddings/jie_gou_she_ji"   # 向量库目录
MODEL_NAME = "BAAI/bge-large-zh"
PAGE_SIZE = 5            # 每页显示多少个 chunk
OUTPUT_FILE = "embe.txt" # 输出文件名


def load_db():
    print("正在加载向量库...")
    emb = HuggingFaceEmbeddings(model_name=MODEL_NAME)
    db = FAISS.load_local(EMB_DIR, emb, allow_dangerous_deserialization=True)
    return db


def dump_all_docs(db):
    print("正在读取全部文档（chunk）...")
    docs = db.similarity_search("", k=2000)  # 足够大，确保拿到全部
    print(f"总 chunk 数量：{len(docs)}")
    return docs


def save_to_file(docs):
    print(f"正在写入 {OUTPUT_FILE} ...")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(f"向量库总 chunk 数量：{len(docs)}\n\n")

        for idx, doc in enumerate(docs):
            f.write(f"\n--- Chunk {idx + 1} ---\n")
            f.write(doc.page_content)
            f.write("\n")

    print(f"写入完成！文件已生成：{OUTPUT_FILE}")


def show_page(docs, page, page_size=PAGE_SIZE):
    total = len(docs)
    total_pages = math.ceil(total / page_size)

    if page < 1:
        page = 1
    if page > total_pages:
        page = total_pages

    start = (page - 1) * page_size
    end = min(start + page_size, total)

    print("\n" + "=" * 60)
    print(f"当前页：{page}/{total_pages}  （共 {total} 个 chunk）")
    print("=" * 60)

    for idx in range(start, end):
        doc = docs[idx]
        print(f"\n--- Chunk {idx + 1} ---")
        content = doc.page_content
        if len(content) > 500:
            content = content[:500] + "..."
        print(content)

    print("\n" + "-" * 60)
    print("输入：n 下一页 | p 上一页 | 数字 跳页 | q 退出")
    print("-" * 60)

    return page


def main():
    db = load_db()
    docs = dump_all_docs(db)

    if not docs:
        print("向量库为空。")
        return

    # 写入 embe.txt
    save_to_file(docs)

    # 分页浏览
    page = 1
    page = show_page(docs, page)

    while True:
        cmd = input(">>> ").strip().lower()

        if cmd == "q":
            print("退出浏览。")
            break
        elif cmd == "n":
            page += 1
        elif cmd == "p":
            page -= 1
        elif cmd.isdigit():
            page = int(cmd)
        else:
            print("无效输入，请输入 n / p / 数字 / q")
            continue

        page = show_page(docs, page)


if __name__ == "__main__":
    main()
