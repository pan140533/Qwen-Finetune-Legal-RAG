import os
import re
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions

# 配置文件路径
TEXT_FILE = "./law_train.txt"          # 已有的语料文件（每行一个句子/段落）
DB_PATH = "./chroma_law_db"            # 向量数据库存储路径
COLLECTION_NAME = "law_texts"

def chunk_text_by_paragraphs(file_path, max_chars=1000):
    """
    将法律文本按段落切块，每个块不超过 max_chars 字符（可调）
    返回 chunks 列表
    """
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    chunks = []
    current_chunk = ""
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if len(current_chunk) + len(line) + 1 <= max_chars:
            if current_chunk:
                current_chunk += "\n" + line
            else:
                current_chunk = line
        else:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = line
    if current_chunk:
        chunks.append(current_chunk)
    return chunks

def main():
    # 1. 加载嵌入模型（轻量，中文效果尚可）
    embed_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    # 自定义 embedding 函数给 chromadb 使用
    class MyEmbeddingFunction(embedding_functions.EmbeddingFunction):
        def __call__(self, texts):
            return embed_model.encode(texts, convert_to_numpy=True).tolist()
    
    # 2. 连接或创建 Chroma 数据库
    chroma_client = chromadb.PersistentClient(path=DB_PATH)
    # 如果已存在集合，先删除（方便重新构建）
    try:
        chroma_client.delete_collection(COLLECTION_NAME)
    except:
        pass
    collection = chroma_client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=MyEmbeddingFunction(),
        metadata={"hnsw:space": "cosine"}
    )
    
    # 3. 切块
    chunks = chunk_text_by_paragraphs(TEXT_FILE, max_chars=800)
    print(f"共生成 {len(chunks)} 个文本块")
    
    # 4. 批量添加（每批 100 条）
    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i:i+batch_size]
        batch_ids = [f"chunk_{j}" for j in range(i, i+len(batch_chunks))]
        collection.add(
            ids=batch_ids,
            documents=batch_chunks,
            metadatas=[{"source": TEXT_FILE} for _ in batch_chunks]
        )
        print(f"已添加 {i+len(batch_chunks)} / {len(chunks)}")
    
    print(f"向量库构建完成，存储路径：{DB_PATH}")

if __name__ == "__main__":
    main()