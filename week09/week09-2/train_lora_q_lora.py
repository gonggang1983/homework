import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training
)

# ======================
# 1. 模型 & 数据配置
# ======================
MODEL = "Qwen/Qwen2.5-Coder-0.5B"
DATA_PATH = "alpaca_data.jsonl"
OUTPUT_DIR = "output_qwen_lora"

# ======================
# 2. 加载数据
# ======================
dataset = load_dataset("json", data_files=DATA_PATH)
train_dataset = dataset["train"]

# ======================
# 3. Tokenizer
# ======================
tokenizer = AutoTokenizer.from_pretrained(
    MODEL,
    trust_remote_code=True
)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# ======================
# 4. 加载 8bit 量化模型
# ======================
print("正在加载 Qwen2.5-Coder 8bit 模型...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    load_in_8bit=True,
    device_map="auto",
    trust_remote_code=True
)

model = prepare_model_for_kbit_training(model)

# ======================
# 5. LoRA 配置（Qwen 专用）
# ======================
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ======================
# 6. 数据预处理（Code Alpaca 格式）
# ======================
def preprocess(examples):
    prompts = []
    for ins, inp in zip(examples["instruction"], examples["input"]):
        if inp.strip():
            prompts.append(
                f"### Instruction:\n{ins}\n\n### Input:\n{inp}\n\n### Response:\n"
            )
        else:
            prompts.append(
                f"### Instruction:\n{ins}\n\n### Response:\n"
            )

    model_inputs = tokenizer(
        prompts,
        max_length=512,
        padding="max_length",
        truncation=True
    )

    labels = tokenizer(
        examples["output"],
        max_length=512,
        padding="max_length",
        truncation=True
    )["input_ids"]

    model_inputs["labels"] = labels
    return model_inputs


train_dataset = train_dataset.map(preprocess, batched=True, remove_columns=train_dataset.column_names)
train_dataset.set_format(type="torch")

# ======================
# 7. 训练参数（4050 6GB）
# ======================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_steps=500,
    save_total_limit=2,
    report_to="none"
)

# ======================
# 8. Trainer
# ======================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer
)

print("开始训练 Qwen2.5-Coder LoRA...")
trainer.train()

print("训练完成，保存模型...")
trainer.save_model(OUTPUT_DIR)
