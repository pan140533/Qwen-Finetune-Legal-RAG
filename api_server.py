from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import chromadb
from sentence_transformers import SentenceTransformer

app = FastAPI()

# ------------------- 加载 RAG 检索器 -------------------
DB_PATH = "./chroma_law_db"
COLLECTION_NAME = "law_texts"
# 使用与构建时相同的嵌入模型
embed_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
chroma_client = chromadb.PersistentClient(path=DB_PATH)
collection = chroma_client.get_collection(COLLECTION_NAME)

def retrieve_law(query: str, top_k=3):
    """检索与查询最相关的法律条文块"""
    query_emb = embed_model.encode([query], convert_to_numpy=True).tolist()
    results = collection.query(query_emb, n_results=top_k)
    documents = results['documents'][0]   # 返回的文本列表
    return documents

# ------------------- 加载微调模型 -------------------
base_model_path = r".\models\Qwen\Qwen2.5-0.5B-Instruct"
lora_path = r".\lora_law_finetuned"

tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)
model = PeftModel.from_pretrained(base_model, lora_path)
model.eval()

# ------------------- 提示模板 -------------------
# 注意：原模型微调任务是“复述法律条文”，现在改为“基于检索到的条文回答问题”
# 为了不重新微调，我们设计一个明确的指令，强制模型仅根据提供的上下文回答。
SYSTEM_PROMPT = """你是一个法律咨询助手。请仅根据下面提供的法律条文内容回答问题。如果条文里没有相关信息，请明确说“根据提供的法律条文无法回答该问题”。"""
def build_rag_prompt(query, retrieved_docs):
    context = "\n\n".join([f"【法律条文】\n{doc}" for doc in retrieved_docs])
    full_prompt = f"""{SYSTEM_PROMPT}

【相关法律条文】
{context}

【用户问题】
{query}

【回答】"""
    return full_prompt

# ------------------- API 定义 -------------------
class QueryRequest(BaseModel):
    query: str
    top_k: int = 3
    max_new_tokens: int = 300

@app.post("/ask")
def ask(request: QueryRequest):
    # 1. 检索相关条文
    docs = retrieve_law(request.query, top_k=request.top_k)
    if not docs:
        return {"answer": "未检索到相关法律条文，请稍后再试。", "retrieved_docs": []}
    
    # 2. 构造 prompt
    prompt = build_rag_prompt(request.query, docs)
    
    # 3. 调用模型生成
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1800).to("cuda")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=request.max_new_tokens,
            temperature=0.2,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # 提取 “【回答】” 后面的内容（可选）
    if "【回答】" in response:
        answer = response.split("【回答】")[-1].strip()
    else:
        answer = response[len(prompt):].strip()
    
    return {
        "answer": answer,
        "retrieved_docs": docs   # 方便调试，可删除
    }

@app.get("/health")
def health():
    return {"status": "ok"}

# 运行方式保持不变：uvicorn api_server:app --host 0.0.0.0 --port 8000