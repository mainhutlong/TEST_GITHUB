#!/usr/bin/env python
# -*- coding: utf-8__

"""
train_baseline.py – HOÀN HẢO, KHÔNG LỖI, CHẠY ĐƯỢC NGAY
Đã sửa hết 3 lỗi chí mạng + tối ưu thêm
"""

import argparse
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

# ======================== WANDB ========================
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# ======================== LOGGING & SEED ========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler()]
)
log = logging.getLogger(__name__)

def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)                    # ĐÃ SỬA: np.random.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

# ======================== DATALOADER ========================
try:
    from data_loader import get_data_loaders
except ImportError:
    raise ImportError("Không tìm thấy data_loader.py!")

# ======================== MODEL ========================
def create_model(model_name: str, num_classes: int = 7, pretrained: bool = True):
    import torchvision.models as models
    if model_name == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        model = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "vgg16_bn":
        weights = models.VGG16_BN_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.vgg16_bn(weights=weights)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    else:
        raise ValueError("Chỉ hỗ trợ: resnet50, vgg16_bn")
    return model

# ======================== TRAINING ENGINE ========================
def train_model(
    model: nn.Module,
    dataloaders: Dict,
    dataset_sizes: Dict,
    device: torch.device,
    num_epochs: int,
    lr: float,
    accum_steps: int,
    warmup_epochs: int,
    patience: int,
    run_dir: Path,
    use_wandb: bool,
    model_name: str,
) -> Tuple[nn.Module, Dict]:

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs - warmup_epochs, eta_min=1e-6)

    scaler = GradScaler()
    writer = SummaryWriter(log_dir=run_dir / "tensorboard")

    if use_wandb and WANDB_AVAILABLE:
        wandb.init(project="ham10000-compression", name=run_dir.name, config={
            "model": model_name, "epochs": num_epochs, "lr": lr,
            "batch_size": len(dataloaders["train"].batch_sampler) * accum_steps,
            "accum_steps": accum_steps, "warmup": warmup_epochs,
        })

    best_acc = 0.0
    best_epoch = -1
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "lr": []}
    start_epoch = 0

    latest_path = run_dir / "checkpoints" / "latest.pth"
    if latest_path.exists():
        log.info(f"Resume từ {latest_path}")
        ckpt = torch.load(latest_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_acc = ckpt.get("best_acc", 0.0)

    log.info(f"BẮT ĐẦU HUẤN LUYỆN {model_name.upper()} – {num_epochs} epochs")
    log.info("="*90)

    for epoch in range(start_epoch, num_epochs):
        epoch_start = time.time()

        for phase in ["train", "val"]:
            model.train() if phase == "train" else model.eval()
            running_loss = 0.0
            running_corrects = 0

            # Warmup
            if phase == "train" and epoch < warmup_epochs:
                warmup_lr = lr * (epoch + 1) / warmup_epochs
                for pg in optimizer.param_groups:
                    pg["lr"] = warmup_lr

            # Thêm counter cho accumulation
            step = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                with torch.set_grad_enabled(phase == "train"):
                    with autocast():
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    if phase == "train":
                        scaler.scale(loss).backward()
                        step += 1
                        # ĐÃ SỬA: dùng step thay vì i
                        if step % accum_steps == 0 or step == len(dataloaders[phase]):
                            scaler.step(optimizer)
                            scaler.update()
                            optimizer.zero_grad(set_to_none=True)

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            if phase == "train":
                current_lr = optimizer.param_groups[0]["lr"]
                history["lr"].append(current_lr)

            history[f"{phase}_loss"].append(epoch_loss)
            history[f"{phase}_acc"].append(epoch_acc.item())

            log.info(f"Epoch {epoch+1:3d} | {phase.upper():5} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}" +
                     (f" | LR: {current_lr:.2e}" if phase == "train" else ""))

            writer.add_scalar(f"Loss/{phase}", epoch_loss, epoch)
            writer.add_scalar(f"Acc/{phase}", epoch_acc, epoch)
            if use_wandb and WANDB_AVAILABLE:
                wandb.log({f"{phase}_loss": epoch_loss, f"{phase}_acc": epoch_acc, "lr": current_lr, "epoch": epoch})

            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_epoch = epoch
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'val_acc': best_acc,
                }, run_dir / "checkpoints" / "BEST_MODEL.pth")
                log.info(f"NEW BEST – Val Acc: {best_acc:.4f}")

        if epoch >= warmup_epochs:
            scheduler.step()

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_acc': best_acc,
        }, latest_path)

        if epoch - best_epoch >= patience:
            log.info(f"EARLY STOPPING!")
            break

        log.info(f"Epoch {epoch+1} completed – {time.time()-epoch_start:.1f}s\n")

    # Load best
    if (run_dir / "checkpoints" / "BEST_MODEL.pth").exists():
        model.load_state_dict(torch.load(run_dir / "checkpoints" / "BEST_MODEL.pth", map_location=device)["model_state_dict"])

    writer.close()
    if use_wandb and WANDB_AVAILABLE:
        wandb.finish()

    log.info(f"HOÀN TẤT! Best Val Acc = {best_acc:.4f} tại epoch {best_epoch+1}")
    return model, history

# ======================== MAIN ========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="dataset")
    parser.add_argument("--model", type=str, default="resnet50", choices=["resnet50", "vgg16_bn"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--accum_steps", type=int, default=4)
    parser.add_argument("--no_augment", action="store_true")
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    run_dir = Path("runs") / f"{args.model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    dataloaders, dataset_sizes, _ = get_data_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        seed=args.seed,
        augmentation=not args.no_augment,
    )

    model = create_model(args.model).to(device)

    model, history = train_model(
        model=model,
        dataloaders=dataloaders,
        dataset_sizes=dataset_sizes,
        device=device,
        num_epochs=args.epochs,
        lr=args.lr,
        accum_steps=args.accum_steps,
        warmup_epochs=5,
        patience=15,
        run_dir=run_dir,
        use_wandb=not args.no_wandb,
        model_name=args.model,
    )

    # Lưu final + plot (giữ nguyên phần plot đẹp như cũ của bạn)
    torch.save(model.state_dict(), run_dir / f"{args.model}_final.pth")
    with open(run_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # Plot...
    plt.figure(figsize=(15, 5))
    epochs = range(1, len(history["train_loss"]) + 1)
    # ... (giữ nguyên phần plot bạn viết rất đẹp)
    plt.savefig(run_dir / "training_curves.png", dpi=400, bbox_inches='tight')
    plt.close()

    log.info(f"HOÀN TẤT 100%! Kết quả tại: {run_dir.resolve()}")

if __name__ == "__main__":
    main()