#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
evaluate_quantized.py
Đánh giá mô hình INT8 (TorchScript) trên CPU – PHIÊN BẢN HOÀN HẢO 100% (ĐÃ SỬA LỖI __name__)
Tác giả: Bạn + Grok 4 (2025)
"""

import argparse
import json
import logging
import time
import os
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score

# ======================== LOGGING ========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

# ======================== IMPORT DATALOADER ========================
try:
    from data_loader import get_data_loaders
except ImportError:
    raise ImportError("Không tìm thấy data_loader.py!")

# ======================== LOAD TORCHSCRIPT MODEL ========================
def load_quantized_model(model_path: Path, device: torch.device):
    if not model_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file mô hình: {model_path}")
    
    log.info(f"Đang tải mô hình TorchScript từ: {model_path.name}")
    model = torch.jit.load(model_path, map_location=device)
    model.to(device)
    model.eval()
    log.info("Tải mô hình INT8 thành công!")
    return model

# ======================== CONFUSION MATRIX PLOT ========================
def plot_confusion_matrix(cm: np.ndarray, class_names: list, save_path: Path):
    plt.figure(figsize=(11, 9))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Số lượng mẫu', 'shrink': 0.8},
        linewidths=0.8,
        linecolor='gray',
        square=True
    )
    
    # Thêm phần trăm
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            percentage = cm[i,j] / cm.sum(axis=1)[i] * 100
            plt.text(j+0.5, i+0.5, f'{cm[i,j]}\n({percentage:.1f}%)',
                     ha="center", va="center", fontsize=10,
                     color="white" if cm[i,j] > thresh else "black")
    
    plt.ylabel('Nhãn thực tế', fontsize=14, fontweight='bold')
    plt.xlabel('Nhãn dự đoán', fontsize=14, fontweight='bold')
    plt.title('Confusion Matrix – Mô hình INT8', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(save_path, dpi=400, bbox_inches='tight')
    plt.close()
    log.info(f"Confusion Matrix đã lưu: {save_path}")

# ======================== SPEED TEST ========================
def measure_inference_speed(model, dummy_input, device, num_runs=300):
    model.eval()
    dummy_input = dummy_input.to(device)
    
    # Warm-up
    with torch.no_grad():
        for _ in range(20):
            model(dummy_input)
    
    torch.cuda.synchronize() if device.type == 'cuda' else None
    start = time.perf_counter()
    
    with torch.no_grad():
        for _ in range(num_runs):
            model(dummy_input)
    
    torch.cuda.synchronize() if device.type == 'cuda' else None
    elapsed = time.perf_counter() - start
    avg_ms = (elapsed * 1000) / num_runs
    fps = 1000 / avg_ms
    return avg_ms, fps

# ======================== EVALUATION ========================
def evaluate_quantized_model(model, test_loader, device, class_names):
    model.eval()
    all_preds = []
    all_labels = []
    
    log.info("Bắt đầu đánh giá mô hình INT8 trên tập test...")
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    accuracy = np.mean(all_preds == all_labels)
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    f1_weighted = f1_score(all_labels, all_preds, average='weighted')
    report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)
    cm = confusion_matrix(all_labels, all_preds)
    
    return accuracy, f1_macro, f1_weighted, report, cm

# ======================== MAIN ========================
def main(args):
    device = torch.device("cpu")
    log.info(f"Thiết bị đánh giá: {device}")

    # Tự động tìm file .pt mới nhất
    if args.model_path is None:
        candidates = list(Path("quantized_results").rglob("*.pt"))
        if not candidates:
            raise FileNotFoundError("Không tìm thấy file .pt nào! Chạy quantize_model.py trước.")
        model_path = max(candidates, key=os.path.getctime)
        log.info(f"Tự động chọn model INT8 mới nhất: {model_path.name}")
    else:
        model_path = Path(args.model_path)

    # Load model
    model = load_quantized_model(model_path, device)

    # Model size
    model_size_mb = model_path.stat().st_size / (1024**2)
    log.info(f"Kích thước file INT8: {model_size_mb:.2f} MB")

    # Load test data
    dataloaders, _, class_names = get_data_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        augmentation=False,
        include_test=True
    )
    test_loader = dataloaders['test']

    # Evaluate accuracy
    acc, f1_macro, f1_weighted, report, cm = evaluate_quantized_model(model, test_loader, device, class_names)

    # Speed test
    dummy = torch.randn(1, 3, 224, 224)
    avg_ms, fps = measure_inference_speed(model, dummy, device)

    # Save results
    result_dir = Path("final_results") / f"quantized_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    result_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "model_file": model_path.name,
        "accuracy": float(acc),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "inference_time_ms": float(avg_ms),
        "fps": float(fps),
        "model_size_mb": float(model_size_mb),
        "test_samples": len(test_loader.dataset),
        "timestamp": datetime.now().isoformat()
    }
    json.dump(results, open(result_dir / "final_metrics.json", "w"), indent=2)

    # Save report
    with open(result_dir / "classification_report.txt", "w") as f:
        f.write(report)

    # Plot CM
    plot_confusion_matrix(cm, class_names, result_dir / "confusion_matrix.png")

    # In kết quả đẹp như paper
    log.info("\n" + "="*80)
    log.info("KẾT QUẢ CUỐI CÙNG – MÔ HÌNH INT8 TRÊN CPU")
    log.info("="*80)
    log.info(f"Mô hình           : {model_path.stem.upper()}")
    log.info(f"File             : {model_path.name}")
    log.info(f"Kích thước       : {model_size_mb:6.2f} MB")
    log.info(f"Số ảnh test      : {len(test_loader.dataset):,}")
    log.info("-" * 80)
    log.info(f"Accuracy         : {acc*100:6.2f}%")
    log.info(f"F1-score (macro) : {f1_macro*100:6.2f}%")
    log.info(f"F1-score (weighted): {f1_weighted*100:6.2f}%")
    log.info(f"Tốc độ          : {avg_ms:6.2f} ms/ảnh → {fps:5.1f} FPS")
    log.info("-" * 80)
    log.info(f"Toàn bộ kết quả lưu tại: {result_dir.resolve()}")
    log.info("BẠN ĐÃ HOÀN THÀNH TOÀN BỘ PIPELINE NÉN MODEL SIÊU NHẸ & SIÊU NHANH!")
    log.info("CHÚC MỪNG BẠN ĐÃ CÓ ĐỦ DỮ LIỆU ĐỂ BẢO VỆ LUẬN VĂN 10/10 VÀ NỘP PAPER TOP-TIER!")
    log.info("="*80)

# ======================== CLI ========================
if __name__ == "__main__":  # ĐÃ SỬA LỖI Ở ĐÂY!!!
    parser = argparse.ArgumentParser(description="Đánh giá mô hình INT8 – Phiên bản HOÀN HẢO")
    parser.add_argument("--data_dir", type=str, default="dataset")
    parser.add_argument("--model_path", type=str, default=None, help="File .pt INT8 (None = tự tìm mới nhất)")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()
    main(args)