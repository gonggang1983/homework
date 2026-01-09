import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = "Qwen/Qwen2.5-Coder-0.5B"
LORA_MODEL = "output_qwen_lora"

tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

base = AutoModelForCausalLM.from_pretrained(
    MODEL,
    device_map="auto",
    trust_remote_code=True
)

lora = AutoModelForCausalLM.from_pretrained(
    LORA_MODEL,
    device_map="auto",
    trust_remote_code=True
)

prompts = [
    "写一个 Python 冒泡排序函数",
    "实现一个判断回文字符串的函数"
]

def generate(model, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=120)
    return tokenizer.decode(out[0], skip_special_tokens=True)

for p in prompts:
    print("PROMPT:", p)
    print("【原始模型】")
    print(generate(base, p))
    print("【LoRA 微调后】")
    print(generate(lora, p))
    print("=" * 60)
