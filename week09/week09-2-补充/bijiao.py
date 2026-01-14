import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

BASE_MODEL = "/modeldata/models/CodeLlama-7b-hf"
QLORA_MODEL = "/modeldata/test/qlora-codellama"

PROMPT = """You are helping a backend engineer clean up user-uploaded paths.

Problem:
Users upload file paths that may contain redundant separators, ".", or "..".
These paths are NOT file system paths and must be processed purely as strings.

Task:
Write a Python function normalize_path(path: str) -> str that:

- Removes redundant slashes
- Resolves "." and ".." correctly
- Preserves whether the path is absolute or relative
- Never allows ".." to escape above the root for absolute paths

Notes:
- This is NOT os.path.normpath
- Do NOT import any library
- The result must be deterministic and safe for production use

Return only the function implementation.
"""

def load_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=False,          # 关键修正：禁用 fast tokenizer
        legacy=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    return tokenizer, model


@torch.no_grad()
def generate(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def evaluate(text):
    return {
        "has_function": bool(re.search(r"def\s+normalize_path", text)),
        "has_placeholder": "Your code here" in text,
        "has_explanation": bool(re.search(r"Explanation|```|###", text)),
        "uses_import": "import " in text,
        "follows_task": (
            "normalize_path" in text and "return" in text
        ),
    }


def run(model_name, model_path):
    print("=" * 80)
    print(model_name)
    print("=" * 80)

    tokenizer, model = load_model(model_path)
    output = generate(model, tokenizer, PROMPT)

    print(output.strip(), "\n")

    metrics = evaluate(output)
    for k, v in metrics.items():
        print(f"{k:20s}: {v}")
    return metrics


def main():
    base_metrics = run("Base Model (CodeLlama-7b-hf)", BASE_MODEL)
    qlora_metrics = run("QLoRA Model (Code Alpaca)", QLORA_MODEL)

    print("\n" + "=" * 80)
    print("Summary Comparison")
    print("=" * 80)

    for key in base_metrics:
        print(
            f"{key:20s} | Base: {base_metrics[key]} | QLoRA: {qlora_metrics[key]}"
        )


if __name__ == "__main__":
    main()
