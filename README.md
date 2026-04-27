# 法律条文 RAG 问答系统

本项目基于 **Qwen2.5-0.5B-Instruct** 模型，通过 **LoRA 微调** + **向量检索增强生成（RAG）** 实现法律条文的问答功能。  
系统会先根据用户问题从法律文本库中检索最相关的条文片段，再将片段作为上下文输入微调模型生成答案。

---

## 项目结构
.
├── 中华人民共和国劳动法_20181229.docx # 原始法律文档

├── 中华人民共和国民法典_20200528.docx # 原始法律文档

├── build_text_corpus.py # 将 docx 转换为纯文本语料（law_train.txt）

├── law_train.txt # 生成的语料（每行一个法律句子/段落）

├── build_vector_store.py # 将语料切块、嵌入并存入 ChromaDB

├── chroma_law_db/ # 向量数据库存储目录（运行后自动生成）

├── train_lora_law.py # LoRA 微调训练脚本

├── lora_law_finetuned/ # 训练好的 LoRA 权重（运行后生成）

├── test_lora_inference.py # 测试微调模型效果（非 RAG，仅复述任务）

├── api_server.py # RAG API 服务（主入口）

└── README.md # 本文件


---

## 功能流程

1. **数据准备**  
   `build_text_corpus.py` 读取两个 `.docx` 文件，提取所有段落，按句号切分并保存到 `law_train.txt`。

2. **LoRA 微调**  
   `train_lora_law.py` 使用 `law_train.txt` 中的法律条文，构建 **“请复述以下法律条文：\n{原文}”** 的指令数据集，对 Qwen2.5-0.5B-Instruct 进行 LoRA 微调。  
   微调后的模型保存在 `lora_law_finetuned` 目录。

3. **构建向量数据库**  
   `build_vector_store.py` 将 `law_train.txt` 中的文本按 800 字符切块，使用 `paraphrase-multilingual-MiniLM-L12-v2` 模型嵌入，存入 ChromaDB（目录 `chroma_law_db`）。

4. **启动 RAG API**  
   `api_server.py` 同时加载：
   - 微调后的 LoRA 模型（用于生成答案）
   - ChromaDB 向量库（用于检索相关条文）  
   对外提供 `/ask` 接口：接收用户问题 → 检索相关条文 → 构造 prompt → 模型生成 → 返回答案。

5. **测试**  
   `test_lora_inference.py` 仅测试微调模型的复述能力（不包含检索），可用于验证 LoRA 训练是否成功。

---

## 环境配置与依赖

### 硬件要求
- **显存**：建议 6GB 以上（使用 4bit 量化 + LoRA 可低至 4GB）
- **内存**：8GB 以上

### 安装依赖

```bash
pip install torch transformers accelerate peft datasets bitsandbytes
pip install docx chromadb sentence-transformers fastapi uvicorn
若需使用 CUDA，请安装对应版本的 torch。

快速开始
1. 准备原始法律文档
将 中华人民共和国劳动法_20181229.docx 和 中华人民共和国民法典_20200528.docx 放在项目根目录。

2. 生成纯文本语料
bash
python build_text_corpus.py
执行后生成 law_train.txt。

3. 训练 LoRA 模型（可选，也可直接使用提供的权重）
bash
python train_lora_law.py
默认只取前 2500 个段落训练，可根据显存调整 all_paragraphs[:2500] 中的数值。
训练完成后，权重保存在 lora_law_finetuned。

4. 构建向量数据库
bash
python build_vector_store.py
执行后生成 chroma_law_db 目录（约数百 MB）。

5. 启动 RAG API 服务
bash
uvicorn api_server:app --host 0.0.0.0 --port 8000
服务启动后，可通过 http://localhost:8000/ask 发送 POST 请求。

6. 调用 API 示例
bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"query": "劳动者怎么解除劳动合同？", "top_k": 3}'
返回示例：

json
{
  "answer": "劳动者提前三十日以书面形式通知用人单位，可以解除劳动合同。",
  "retrieved_docs": ["劳动者提前三十日以书面形式通知用人单位，可以解除劳动合同。", ...]
}
top_k 控制检索到的相关条文数量，max_new_tokens 控制生成回答的最大长度。

自定义与改进建议
替换嵌入模型：修改 build_vector_store.py 和 api_server.py 中的 SentenceTransformer 模型名称，例如使用 "BAAI/bge-large-zh" 获得更好的中文效果。

调整切块大小：修改 chunk_text_by_paragraphs 中的 max_chars 参数（推荐 500~1000）。

更换基座模型：将 base_model_path 改为其他 HuggingFace 模型路径（如 Qwen/Qwen2.5-7B-Instruct），需注意显存。

增加 rerank 步骤：在检索后加入重排序模型（如 BAAI/bge-reranker-base）提高精度。

前端界面：可基于 FastAPI 的静态文件服务或另写前端，调用 /ask 接口展示问答界面。

常见问题
1. ChromaDB 报错 No module named 'chromadb'
请执行 pip install chromadb。

2. 训练时显存不足
减小 max_length（如 256）

减小 per_device_train_batch_size 或增加 gradient_accumulation_steps

减少训练段落数 all_paragraphs[:500]

3. 检索总是返回空结果
检查 build_vector_store.py 是否正确运行，以及 DB_PATH 路径是否一致。

4. 模型回答没有使用检索到的条文
检查 api_server.py 中的 SYSTEM_PROMPT 是否强调“仅根据提供的法律条文回答”，以及 prompt 中是否正确拼接了 context。

许可证
本项目中的代码采用 MIT 许可证。
法律文本（劳动法、民法典）归国家公开信息所有，仅用于演示目的。

致谢
Qwen 团队提供的基座模型

HuggingFace 的 Transformers 库

Chroma 向量数据库

text

这个 README 涵盖了从数据准备到 API 调用的完整说明，适合作为项目文档。你可以根据实际存放路径和模型名称微调。
