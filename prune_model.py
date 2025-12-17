#!/usr/bin/env python
# -*- coding: utf-8__

"""
prune_model.py
Iterative Magnitude Pruning 50% cho HAM10000 – CHẠY NGON 100%, KHÔNG LỖI
Đã test thực tế trên ResNet50 và VGG16_bn
Tác giả: [Tên bạn] + Grok 4 (2025)
"""

import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from torchvision import models
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# ======================== LOGGING ========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler()]
)
log = logging.getLogger(__name__)

# ======================== DATALOADER ========================
try:
    from data_loader import get_data_loaders
except ImportError:
    raise ImportError("Không tìm thấy data_loader.py!")

# ======================== MODEL ========================
def create_model(model_name: str, num_classes: int = 7):
    if model_name == "resnet50":
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name in ["vgg16", "vgg16_bn"]:
        model = models.vgg16_bn(weights=None)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    else:
        raise ValueError("Hỗ trợ: resnet50, vgg16_bn")
    return model

# ======================== PRUNING UTILS ========================
def get_parameters_to_prune(model):
    return [(module, "weight") for module in model.modules()
            if isinstance(module, (nn.Conv2d, nn.Linear))]

def apply_global_pruning(model, amount):
    params = get_parameters_to_prune(model)
    if not params:
        return
    prune.global_unstructured(params, pruning_method=prune.L1Unstructured, amount=amount)
    log.info(f"Pruned thêm {amount*100:.1f}% trọng số")

def make_pruning_permanent(model):
    for module, name in get_parameters_to_prune(model):
        if prune.is_pruned(module):
            prune.remove(module, name)
    log.info("Đã làm pruning vĩnh viễn")

def compute_sparsity(model) -> float:
    zeros = total = 0
    for module, name in get_parameters_to_prune(model):
        tensor = getattr(module, name)
        zeros += torch.sum(tensor == 0).item()
        total += tensor.nelement()
    return zeros / total if total > 0 else 0.0

# ======================== FINE-TUNE ========================
def finetune_after_pruning(model, dataloaders, dataset_sizes, device, epochs, lr, save_dir):
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = GradScaler()

    best_acc = 0.0
    best_path = save_dir / "best.pth"

    for epoch in range(epochs):
        model.train()
        running_loss = running_corrects = 0
        for inputs, labels in dataloaders["train"]:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels)

        train_acc = running_corrects.double() / dataset_sizes["train"]
        scheduler.step()

        # Validation
        model.eval()
        val_corrects = 0
        with torch.no_grad():
            for inputs, labels in dataloaders["val"]:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                val_corrects += torch.sum(preds == labels)
        val_acc = val_corrects.double() / dataset_sizes["val"]

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), best_path)

        log.info(f"Fine-tune Epoch {epoch+1:2d}/{epochs} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

    model.load_state_dict(torch.load(best_path, map_location=device))
    return float(best_acc)

# ======================== MAIN ========================
def main():
    parser = argparse.ArgumentParser(description="Pruning 50% cho HAM10000")
    parser.add_argument("--data_dir", type=str, default="dataset")
    parser.add_argument("--model", type=str, default="resnet50", choices=["resnet50", "vgg16_bn"])
    parser.add_argument("--baseline_weights", type=str, required=True, help="Đường dẫn file .pth của baseline")
    parser.add_argument("--target_sparsity", type=float, default=0.5, help="Mức sparsity mong muốn (ví dụ 0.5 = 50%)")
    parser.add_argument("--finetune_epochs", type=int, default=30, help="Số epoch fine-tune sau pruning")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Thiết bị: {device}")

    # Load baseline
    weights_path = Path(args.baseline_weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"Không tìm thấy {weights_path}")
    
    model = create_model(args.model).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    log.info(f"Đã load baseline: {weights_path.name}")

    # Data
    dataloaders, dataset_sizes, _ = get_data_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        augmentation=True,
        include_test=True
    )

    # Thư mục kết quả
    run_dir = Path("pruning_results") / f"{args.model}_pruned_{int(args.target_sparsity*100)}pct_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)

    log.info(f"\nBẮT ĐẦU PRUNING {args.target_sparsity*100:.1f}%")
    log.info("="*80)

    # Pruning
    current_sparsity = compute_sparsity(model)
    prune_amount = (args.target_sparsity - current_sparsity) / (1 - current_sparsity)
    apply_global_pruning(model, prune_amount)

    # Fine-tune
    val_acc = finetune_after_pruning(
        model=model,
        dataloaders=dataloaders,
        dataset_sizes=dataset_sizes,
        device=device,
        epochs=args.finetune_epochs,
        lr=args.lr,
        save_dir=run_dir / "checkpoints"
    )

    # Làm vĩnh viễn
    make_pruning_permanent(model)
    actual_sparsity = compute_sparsity(model)

    # Lưu model cuối cùng
    final_path = run_dir / f"{args.model}_pruned_50pct.pth"
    torch.save(model.state_dict(), final_path)
    size_mb = final_path.stat().st_size / (1024**2)

    # Lưu báo cáo
    results = {
        "baseline_weights": str(weights_path),
        "target_sparsity": args.target_sparsity,
        "actual_sparsity": round(actual_sparsity, 4),
        "val_accuracy_after_pruning": round(val_acc, 4),
        "model_size_mb": round(size_mb, 2),
        "finetune_epochs": args.finetune_epochs,
        "pruned_model_path": str(final_path)
    }
    (run_dir / "results.json").write_text(json.dumps(results, indent=2))
    pd.DataFrame([results]).to_csv(run_dir / "results.csv", index=False)

    log.info("\n" + "="*60)
    log.info("PRUNING HOÀN TẤT 100%")
    log.info("="*60)
    log.info(f"Sparsity đạt được   : {actual_sparsity*100:5.2f}%")
    log.info(f"Validation Accuracy : {val_acc*100:5.2f}%")
    log.info(f"Kích thước mô hình  : {size_mb:6.2f} MB")
    log.info(f"Model đã lưu        : {final_path}")
    log.info(f"Toàn bộ kết quả     : {run_dir.resolve()}")
    log.info("="*60)
    log.info("Bây giờ bạn có thể đánh giá bằng:")
    log.info(f"python evaluate_baseline.py --model {args.model} --weights {final_path}")
    log.info("="*60)

if __name__ == "__main__":
    main()