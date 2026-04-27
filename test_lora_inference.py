import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 基础模型路径（和训练时一致）
base_model_path = r".\models\Qwen\Qwen2.5-0.5B-Instruct"
# LoRA 权重保存路径
lora_path = r".\lora_law_finetuned"

# 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)

# 加载基础模型（可以用 fp16 以节省显存，如果不做量化）
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

# 加载 LoRA 权重
model = PeftModel.from_pretrained(base_model, lora_path)

# 测试几个 prompt
test_prompts = [
    "请复述以下法律条文：\n劳动者提前三十日以书面形式通知用人单位，可以解除劳动合同。",
    "请复述以下法律条文：\n用人单位自用工之日起即与劳动者建立劳动关系。",
    "请复述以下法律条文：\n有下列情形之一的，婚姻无效：（一）重婚；（二）有禁止结婚的亲属关系；（三）未到法定婚龄。"
]

for prompt in test_prompts:
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        temperature=0.1,
        do_sample=False
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("输入:", prompt)
    print("输出:", response)
    print("-" * 50)