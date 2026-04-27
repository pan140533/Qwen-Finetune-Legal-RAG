
import os
import torch
from docx import Document
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType

# -------------------- 配置路径 --------------------
DOC_PATHS = [
    r".\中华人民共和国劳动法_20181229.docx",
    r".\中华人民共和国民法典_20200528.docx"
]
MODEL_PATH = r".\models\Qwen\Qwen2.5-0.5B-Instruct"
OUTPUT_DIR = r"./lora_law_finetuned"

# -------------------- 1. 数据准备：从 docx 提取文本 --------------------
def extract_text_from_docx(path):
    """读取 docx 文件，返回所有段落文本（去除空行和短句）"""
    doc = Document(path)
    paragraphs = []
    for para in doc.paragraphs:
        text = para.text.strip()
        # 过滤掉空行和太短的句子（可能是标题或编号）
        if len(text) > 5:
            paragraphs.append(text)
    return paragraphs

all_paragraphs = []
for doc_path in DOC_PATHS:
    print(f"正在读取: {doc_path}")
    paras = extract_text_from_docx(doc_path)
    all_paragraphs.extend(paras)
print(f"共提取到 {len(all_paragraphs)} 个段落")

# 为了快速演示，只取前 50 个段落（可根据需要调整）
all_paragraphs = all_paragraphs[:2500]

# 构建指令数据集：每个样本包含指令和输出
# 这里使用最简单的复述任务，你可以替换为自己的问答数据
def build_dataset(paragraphs):
    data = []
    for para in paragraphs:
        # 构造输入：给模型一个指令，要求复述法律条文
        instruction = "请复述以下法律条文："
        # 将指令和原文拼接作为输入（模型需学习输出原文）
        # 注意：对于因果语言模型，我们需要将指令+原文作为输入，但训练时要求模型生成原文部分
        # 更好的做法是使用模板：<指令>\n{原文}，并设置 labels 仅对原文部分计算 loss。
        # 但为了简单，这里直接使用完整文本作为输入，labels 也设为完整文本，让模型学习复述。
        # 实际应用中，应该区分 prompt 和 response，但作为演示，这也能工作。
        text = f"{instruction}\n{para}"
        data.append({"text": text})
    return data

dataset_dict = build_dataset(all_paragraphs)
dataset = Dataset.from_list(dataset_dict)

# -------------------- 2. 加载 tokenizer 和模型（4bit 量化） --------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
# 设置 pad_token（Qwen 可能没有默认 pad_token）
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 量化配置，大幅降低显存占用
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

# -------------------- 3. 配置 LoRA --------------------
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,                # 低秩矩阵的秩
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Qwen 常见模块名
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  # 应该只显示少量参数

# -------------------- 4. 数据预处理：tokenize --------------------
def tokenize_function(examples):
    # 对文本进行 tokenize，同时生成 labels（与 input_ids 相同，因为是语言建模）
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        padding=False,
        max_length=512,          # 根据显存调整，太长会 OOM
        return_tensors=None
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

tokenized_dataset = dataset.map(tokenize_function, batched=False, remove_columns=["text"])

# -------------------- 5. 训练参数配置 --------------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,               # 小 batch 防爆显存
    gradient_accumulation_steps=8,                # 模拟 batch_size=8 的效果
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=True,                                    # 混合精度
    optim="paged_adamw_8bit",                     # 8bit 优化器进一步省显存
    logging_steps=10,
    save_strategy="epoch",
    save_total_limit=2,
    remove_unused_columns=False,
    report_to="none"                              # 禁用 wandb 等
)

# 数据收集器，用于动态 padding（节省显存）
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,        # 因果语言模型不需要 MLM
    pad_to_multiple_of=8
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

# -------------------- 6. 开始训练 --------------------
print("开始训练...")
trainer.train()

# -------------------- 7. 保存 LoRA 权重 --------------------
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"LoRA 权重已保存至: {OUTPUT_DIR}")

# -------------------- 8. 测试推理（加载 LoRA 并尝试生成） --------------------
print("\n测试微调效果：")
from peft import PeftModel

# 重新加载基础模型（不量化以便生成，也可以保持量化）
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True
)
lora_model = PeftModel.from_pretrained(base_model, OUTPUT_DIR)

# 测试 prompt
test_prompt = "请复述以下法律条文：\n劳动者提前三十日以书面形式通知用人单位，可以解除劳动合同。"
inputs = tokenizer(test_prompt, return_tensors="pt").to("cuda")
outputs = lora_model.generate(
    **inputs,
    max_new_tokens=100,
    temperature=0.1,
    do_sample=False
)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("输入:", test_prompt)
print("输出:", response)

# 你也可以尝试用自己的问答对测试