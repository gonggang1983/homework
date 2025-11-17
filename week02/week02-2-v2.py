import os
import argparse
import random

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split


# =======================
# 1. 手写 transform 组件
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
        # size: (w, h) 或 int
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


class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img: Image.Image):
        if random.random() < self.p:
            return img.transpose(Image.FLIP_LEFT_RIGHT)
        return img


class RandomResizedCrop:
    """
    简化版 RandomResizedCrop:
    先 Resize 到较大尺寸，再在里面随机裁一个子块，最后缩放到目标尺寸。
    """
    def __init__(self, size, scale=(0.8, 1.0)):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size
        self.scale = scale

    def __call__(self, img: Image.Image):
        w, h = img.size
        area = w * h

        for _ in range(10):
            target_area = random.uniform(*self.scale) * area
            aspect_ratio = random.uniform(3. / 4., 4. / 3.)

            new_w = int(round((target_area * aspect_ratio) ** 0.5))
            new_h = int(round((target_area / aspect_ratio) ** 0.5))

            if new_w <= w and new_h <= h:
                x1 = random.randint(0, w - new_w)
                y1 = random.randint(0, h - new_h)
                img = img.crop((x1, y1, x1 + new_w, y1 + new_h))
                return img.resize(self.size, Image.BILINEAR)

        # 回退：如果上面没成功，就中心裁剪再 resize
        img = CenterCrop(min(w, h))(img)
        return img.resize(self.size, Image.BILINEAR)


class ToTensor:
    def __call__(self, img: Image.Image):
        # PIL (H,W,C) -> Tensor (C,H,W), 归一化到 [0,1]
        arr = np.array(img, dtype=np.float32) / 255.0
        if arr.ndim == 2:  # 灰度图情况（基本不会发生）
            arr = np.stack([arr, arr, arr], axis=-1)
        arr = arr.transpose(2, 0, 1)
        return torch.from_numpy(arr)


class Normalize:
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)

    def __call__(self, tensor: torch.Tensor):
        return (tensor - self.mean) / self.std


# =======================
# 2. 数据集定义
# =======================

class CarBrandDataset(Dataset):
    """
    通过 re_id_1000_train.txt / re_id_1000_test.txt 加载数据，
    使用路径的顶层目录 1~10 表示类别，映射为 0~9。
    """

    def __init__(self, txt_file, img_root, transform=None):
        self.img_root = img_root
        self.transform = transform
        self.samples = []

        missing_count = 0

        with open(txt_file, "r", encoding="utf-8") as f:
            for line in f:
                rel_path = line.strip()
                if not rel_path:
                    continue

                rel_path_norm = rel_path.replace("\\", os.sep)
                parts = rel_path_norm.split(os.sep)

                top_dir = int(parts[0])   # 1~10
                label = top_dir - 1       # 0~9

                img_path = os.path.join(self.img_root, rel_path_norm)

                if os.path.exists(img_path):
                    self.samples.append((img_path, label))
                else:
                    missing_count += 1

        if missing_count > 0:
            print(f"[WARN] {txt_file} 中有 {missing_count} 个样本找不到对应图片，已自动跳过。")
        if len(self.samples) == 0:
            raise RuntimeError(f"No valid samples found in {txt_file}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        return img, label


# =======================
# 3. ResNet18 纯 PyTorch 实现
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

        # 初始化（模仿 torchvision）
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

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
# 4. 训练 & 验证函数
# =======================

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for imgs, labels in loader:
        imgs = imgs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


@torch.no_grad()
def eval_one_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for imgs, labels in loader:
        imgs = imgs.to(device)
        labels = labels.to(device)

        outputs = model(imgs)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


# =======================
# 5. 主函数
# =======================

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 1) 定义数据增强
    train_transform = Compose([
        Resize((256, 256)),
        RandomResizedCrop(224, scale=(0.8, 1.0)),
        RandomHorizontalFlip(p=0.5),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225]),
    ])

    test_transform = Compose([
        Resize((256, 256)),
        CenterCrop(224),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225]),
    ])

    # 2) 构建数据集
    full_train_dataset = CarBrandDataset(args.train_txt, args.img_root, transform=train_transform)
    test_dataset = CarBrandDataset(args.test_txt, args.img_root, transform=test_transform)

    # 从 train 再划一部分做验证集
    val_size = int(args.val_ratio * len(full_train_dataset))
    train_size = len(full_train_dataset) - val_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

    print(f"Train samples: {train_size}, Val samples: {val_size}, Test samples: {len(test_dataset)}")

    # 3) DataLoader
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=args.num_workers)

    # 4) 模型、损失、优化器
    model = resnet18(num_classes=10).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=args.lr_step_size,
        gamma=args.lr_gamma
    )

    best_val_acc = 0.0
    best_state = None

    # 5) 训练循环
    for epoch in range(args.epochs):
        print(f"\nEpoch [{epoch + 1}/{args.epochs}]")

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = eval_one_epoch(model, val_loader, criterion, device)

        scheduler.step()

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f}, Val   Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict()

    print(f"\nBest Val Acc: {best_val_acc:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    # 6) 测试集评估
    test_loss, test_acc = eval_one_epoch(model, test_loader, criterion, device)
    print(f"\nTest  Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

    if test_acc >= 0.80:
        print("✅ 测试集准确率 >= 80%，满足作业要求。")
    else:
        print("⚠️ 测试集准确率 < 80%，建议调整学习率、epoch 或数据增强后重训。")

    # 7) 保存模型
    if args.save_model:
        os.makedirs(os.path.dirname(args.save_model), exist_ok=True)
        torch.save(model.state_dict(), args.save_model)
        print(f"Model saved to {args.save_model}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--img_root", type=str, required=True,
                        help="图片根目录，比如 image")
    parser.add_argument("--train_txt", type=str, required=True,
                        help="训练索引文件，比如 re_id_1000_train.txt")
    parser.add_argument("--test_txt", type=str, required=True,
                        help="测试索引文件，比如 re_id_1000_test.txt")

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--lr_step_size", type=int, default=7)
    parser.add_argument("--lr_gamma", type=float, default=0.1)

    parser.add_argument("--val_ratio", type=float, default=0.1,
                        help="从训练集划分验证集的比例")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="DataLoader 的工作线程数，Windows 建议从 0 开始试")

    parser.add_argument("--save_model", type=str, default="checkpoints/car_brand_resnet18_no_tv.pth")

    args = parser.parse_args()
    main(args)
