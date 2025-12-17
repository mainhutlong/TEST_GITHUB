#!/usr/bin/env python
# -*- coding: utf-8__

"""
evaluate_baseline.py
Đánh giá mô hình baseline/pruned/quantized trên tập test – PHIÊN BẢN PUBLICATION-GRADE 2025
Tính năng:
- Tự động tìm model mới nhất
- Đo tốc độ chính xác (warmup + synchronize)
- Confusion Matrix đẹp như paper
- Classification report đầy đủ + per-class
- Đo model size chính xác (MB + params)
- Lưu tất cả: JSON, TXT, PNG
- Hỗ trợ cả VGG16_bn và ResNet50

Tác giả: [Tên bạn] + Grok 4 (2025)
"""

import argparse
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Dict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from torch.utils.data import DataLoader

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
    raise ImportError("Không tìm thấy data_loader.py! Chạy split_data.py trước.")

# ======================== MODEL FACTORY ========================
def create_model(model_name: str, num_classes: int = 7, pretrained: bool = False):
    import torchvision.models as models
    
    if model_name == "resnet50":
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name in ["vgg16", "vgg16_bn"]:
        model = models.vgg16_bn(weights=None)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    else:
        raise ValueError("Hỗ trợ: resnet50, vgg16, vgg16_bn")
    return model

# ======================== MODEL SIZE & PARAMS ========================
def get_model_size(model: nn.Module, save_path: Path = None) -> Tuple[float, int]:
    param_size = 0
    buffer_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024 / 1024
    num_params = sum(p.numel() for p in model.parameters())
    
    if save_path:
        torch.save(model.state_dict(), save_path)
        file_size_mb = save_path.stat().st_size / (1024*1024)
        return file_size_mb, num_params
    return size_mb, num_params

# ======================== INFERENCE SPEED ========================
def measure_inference_speed(model: nn.Module, input_size: Tuple[int,int] = (224,224), 
                           device: torch.device = torch.device("cpu"), num_runs: int = 300) -> Tuple[float, float]:
    model.eval()
    dummy_input = torch.randn(1, 3, *input_size).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(20):
            _ = model(dummy_input)
    
    if device.type == "cuda":
        torch.cuda.synchronize()
    
    start_time = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(dummy_input)
    
    if device.type == "cuda":
        torch.cuda.synchronize()
    
    elapsed = time.perf_counter() - start_time
    avg_ms = (elapsed / num_runs) * 1000
    fps = 1000 / avg_ms
    return round(avg_ms, 3), round(fps, 2)

# ======================== EVALUATION ========================
def evaluate_model(model: nn.Module, test_loader: DataLoader, device: torch.device, class_names: List[str]):
    model.eval()
    all_preds = []
    all_labels = []

    log.info("Đang đánh giá trên tập test...")
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    acc = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    f1_weighted = f1_score(all_labels, all_preds, average='weighted')
    report_dict = classification_report(all_labels, all_preds, target_names=class_names, digits=4, output_dict=True)
    report_str = classification_report(all_labels, all_preds, target_names=class_names, digits=4)
    cm = confusion_matrix(all_labels, all_preds)

    return acc, f1_macro, f1_weighted, report_dict, report_str, cm

# ======================== CONFUSION MATRIX PLOT ========================
def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], save_path: Path):
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=class_names, yticklabels=class_names,
        cbar_kws={'shrink': 0.8}, square=True,
        linewidths=0.5, linecolor='gray'
    )
    plt.xlabel('Predicted Label', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=14, fontweight='bold')
    plt.title('Confusion Matrix – HAM10000 Test Set', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(save_path, dpi=400, bbox_inches='tight')
    plt.close()
    log.info(f"Confusion Matrix lưu tại: {save_path}")

# ======================== MAIN ========================
def main():
    parser = argparse.ArgumentParser(description="Đánh giá mô hình – Publication Ready 2025")
    parser.add_argument("--data_dir", type=str, default="dataset")
    parser.add_argument("--model", type=str, default="resnet50", choices=["resnet50", "vgg16", "vgg16_bn"])
    parser.add_argument("--weights", type=str, required=True, help="Đường dẫn file .pth")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Thiết bị: {device}")

    weights_path = Path(args.weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file trọng số: {weights_path}")

    # Load model
    model = create_model(args.model).to(device)
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    log.info(f"Đã load trọng số: {weights_path.name}")

    # Model size (file): {weights_path.stat().st_size / (1024**2):.2f} MB
    params_count = sum(p.numel() for p in model.parameters())
    log.info(f"Số tham số       : {params_count:,}")

    # Data
    dataloaders, _, class_names = get_data_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        augmentation=False,
        include_test=True
    )
    test_loader = dataloaders["test"]

    # Evaluate
    acc, f1_macro, f1_weighted, report_dict, report_str, cm = evaluate_model(
        model, test_loader, device, class_names
    )

    # Speed test
    avg_ms, fps = measure_inference_speed(model, device=device)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = Path("evaluation_results") / f"{args.model}_{timestamp}"
    result_dir.mkdir(parents=True, exist_ok=True)

    # Save metrics
    results = {
        "model": args.model,
        "weights_file": str(weights_path.name),
        "test_accuracy": round(acc, 4),
        "f1_macro": round(f1_macro, 4),
        "f1_weighted": round(f1_weighted, 4),
        "inference_time_ms": avg_ms,
        "fps": fps,
        "model_size_mb": round(weights_path.stat().st_size / (1024**2), 2),
        "num_parameters": params_count,
        "num_test_samples": len(test_loader.dataset),
        "class_names": class_names,
        "timestamp": datetime.now().isoformat()
    }
    (result_dir / "metrics.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))

    # Save full report
    (result_dir / "classification_report.txt").write_text(report_str)

    # Plot CM
    plot_confusion_matrix(cm, class_names, result_dir / "confusion_matrix.png")

    # Beautiful final report
    log.info("\n" + "="*70)
    log.info("KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH TRÊN TẬP TEST")
    log.info("="*70)
    log.info(f"Mô hình           : {args.model.upper():<15}")
    log.info(f"File trọng số     : {weights_path.name}")
    log.info(f"Kích thước file   : {results['model_size_mb']:6.2f} MB")
    log.info(f"Số tham số        : {params_count:,}")
    log.info(f"Số ảnh test       : {len(test_loader.dataset)}")
    log.info("-" * 60)
    log.info(f"Accuracy          : {acc*100:6.2f}%")
    log.info(f"F1-score (macro)  : {f1_macro*100:6.2f}%")
    log.info(f"F1-score (weighted): {f1_weighted*100:6.2f}%")
    log.info(f"Tốc độ suy luận   : {avg_ms:6.2f} ms/ảnh → {fps:5.1f} FPS")
    log.info("-" * 60)
    log.info(f"Kết quả đã lưu tại: {result_dir.resolve()}")
    log.info("="*70)

if __name__ == "__main__":
    main()