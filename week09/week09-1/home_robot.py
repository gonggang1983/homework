import torch
from PIL import Image
from transformers import (
    LlavaNextProcessor,
    LlavaNextForConditionalGeneration
)
import sys
import os


class HomeRobot:
    """
    使用 LLaVA-NeXT 完成：
    1. 场景理解
    2. 目标定位（语言描述）
    3. 视觉问答（VQA）
    """

    def __init__(
        self,
        model_name="E:/download/llava-v1.6-mistral-7b-hf",
        device=None,
        hf_token=None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        print(f"[INFO] Loading LLaVA-NeXT on {self.device}")

        self.processor = LlavaNextProcessor.from_pretrained(
            model_name,
            token=hf_token
        )

        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            low_cpu_mem_usage=True,
            token=hf_token
        )

        self.model.to(self.device)
        self.model.eval()

        print("[INFO] Model loaded successfully")

    @torch.no_grad()
    def ask(self, image: Image.Image, question: str, max_new_tokens=128):
        """
        统一的视觉问答接口
        """
        prompt = f"<image>\nUSER: {question}\nASSISTANT:"

        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt"
        ).to(self.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False
        )

        answer = self.processor.decode(
            outputs[0],
            skip_special_tokens=True
        )

        # 只保留 Assistant 的回答部分
        if "ASSISTANT:" in answer:
            answer = answer.split("ASSISTANT:")[-1].strip()

        return answer

    def run(self, image_path: str):
        if not os.path.exists(image_path):
            raise FileNotFoundError(image_path)

        image = Image.open(image_path).convert("RGB")

        print("\n=== Scene Understanding ===")
        scene_desc = self.ask(
            image,
            "详细描述一下图中的场景。"
        )
        print(scene_desc)

        print("\n=== Object Grounding (Language-based) ===")
        grounding_desc = self.ask(
            image,
            "列出图中主要的物体并统计数量及其位置。"
        )
        print(grounding_desc)

        print("\n=== Visual Question Answering ===")
        vqa_answer = self.ask(
            image,
            "在这个场景里最重要的是什么物体？为什么？"
        )
        print(vqa_answer)


def parse_hf_token():
    """
    支持：
    python home_robot.py image.jpg token=hf_xxx
    """
    for arg in sys.argv:
        if arg.startswith("token="):
            return arg.split("=", 1)[1]
    return None


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python home_robot.py <image_path> [token=hf_xxx]")
        sys.exit(1)

    image_path = sys.argv[1]
    hf_token = parse_hf_token()

    robot = HomeRobot(hf_token=hf_token)
    robot.run(image_path)
