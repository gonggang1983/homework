import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model

MODEL_NAME = "/modeldata/models/CodeLlama-7b-hf"
DATA_PATH = "/modeldata/test/code_alpaca_20k.json"
dataset = load_dataset(
    "json",
    data_files=DATA_PATH
)["train"]

# --------------------
# Tokenizer
# --------------------
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    use_fast=False
)
tokenizer.pad_token = tokenizer.eos_token

# --------------------
# QLoRA config
# --------------------
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# --------------------
# Model
# --------------------
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
)
model.config.use_cache = False

# --------------------
# LoRA config
# --------------------
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# --------------------
# Dataset
# --------------------
dataset = load_dataset("json", data_files=DATA_PATH)

def format_prompt(example):
    prompt = f"""### Instruction:
{example["instruction"]}

### Input:
{example["input"]}

### Response:
{example["output"]}"""
    return prompt

def tokenize_fn(example):
    text = format_prompt(example)
    tokens = tokenizer(
        text,
        truncation=True,
        max_length=512,
        padding="max_length",
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

dataset = dataset["train"].map(
    tokenize_fn,
    remove_columns=dataset["train"].column_names,
    num_proc=4,
)

# --------------------
# Training args
# --------------------
training_args = TrainingArguments(
    output_dir="./qlora-codellama",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    num_train_epochs=2,
    fp16=True,
    logging_steps=20,
    save_strategy="epoch",
    save_total_limit=2,
    optim="paged_adamw_8bit",
    report_to="none",
)

# --------------------
# Trainer
# --------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

trainer.train()
model.save_pretrained("./qlora-codellama")
tokenizer.save_pretrained("./qlora-codellama")
