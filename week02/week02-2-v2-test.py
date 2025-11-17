import argparse
import os
import random

import numpy as np
from PIL import Image

import torch
import torch.nn as nn


# =======================
# 1. 手写 transforms
# =======================

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img


class Resize:
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size

    def __call__(self, img: Image.Image):
        return img.resize(self.size, Image.BILINEAR)


class CenterCrop:
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size

    def __call__(self, img: Image.Image):
        w, h = img.size
        tw, th = self.size
        i = int(round((h - th) / 2.0))
        j = int(round((w - tw) / 2.0))
        return img.crop((j, i, j + tw, i + th))


class ToTensor:
    def __call__(self, img: Image.Image):
        arr = np.array(img, dtype=np.float32) / 255.0  # HWC
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        arr = arr.transpose(2, 0, 1)  # CHW
        return torch.from_numpy(arr)


class Normalize:
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)

    def __call__(self, tensor: torch.Tensor):
        return (tensor - self.mean) / self.std


# =======================
# 2. ResNet18 纯 PyTorch
# =======================

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super().__init__()
        self.inplanes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64,  layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion

        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


def resnet18(num_classes=10):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)


# =======================
# 3. 类别映射
# =======================

id2name = {
    0: "丰田_凯美瑞 / Toyota_Camry",
    1: "丰田_卡罗拉 / Toyota_Corolla",
    2: "丰田_花冠 / Toyota_Corolla_EX",
    3: "别克_君越 / Buick_LaCrosse",
    4: "大众_迈腾 / Volkswagen_Magotan",
    5: "奥迪_A4 / Audi_A4",
    6: "日产_轩逸 / Nissan_Sylphy",
    7: "日产_骐达 / Nissan_Tiida",
    8: "本田_雅阁 / Honda_Accord",
    9: "福特_福克斯 / Ford_Focus",
}


# =======================
# 4. 推理函数
# =======================

def build_transform():
    return Compose([
        Resize((256, 256)),
        CenterCrop(224),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225]),
    ])


def load_model(model_path, device):
    model = resnet18(num_classes=10)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return model


def predict(model, img_path, device):
    transform = build_transform()

    img = Image.open(img_path).convert("RGB")
    img = transform(img)          # CxHxW tensor
    img = img.unsqueeze(0).to(device)  # 1xCxHxW

    with torch.no_grad():
        outputs = model(img)
        probs = torch.softmax(outputs, dim=1)
        conf, pred = probs.max(1)

    pred_id = int(pred.item())
    conf = float(conf.item())
    name = id2name.get(pred_id, f"class_{pred_id}")
    return pred_id, name, conf


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                        help="模型权重 .pth 路径（训练脚本保存的文件）")
    parser.add_argument("--image", type=str, required=True,
                        help="要预测的图片路径")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    if not os.path.exists(args.model):
        raise FileNotFoundError(f"模型文件不存在: {args.model}")
    if not os.path.exists(args.image):
        raise FileNotFoundError(f"图片文件不存在: {args.image}")

    model = load_model(args.model, device)
    pred_id, name, conf = predict(model, args.image, device)

    print(f"预测类别ID: {pred_id}")
    print(f"预测品牌: {name}")
    print(f"置信度: {conf:.4f}")
