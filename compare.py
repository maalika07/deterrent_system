"""
Model Comparison — ResNet50 vs EfficientNet-B0 vs MobileNetV3-Large
Computes every standard classification metric and saves publication-ready plots.

Usage:
    python compare_models.py

Requires:
    animal_classifier_resnet50.pth
    animal_classifier_efficientnet_b0.pth
    animal_classifier_mobilenet_v3.pth

Saved outputs:
    confusion_matrix_resnet50.png
    confusion_matrix_efficientnet_b0.png
    confusion_matrix_mobilenet_v3.png
    per_class_metrics_resnet50.png
    per_class_metrics_efficientnet_b0.png
    per_class_metrics_mobilenet_v3.png
    roc_curves.png
    comparison_bar_chart.png
    metrics_heatmap.png
    speed_accuracy.png
    metrics_comparison.csv
    metrics_report.txt
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, balanced_accuracy_score,
    roc_auc_score, roc_curve, auc,
    precision_score, recall_score, f1_score,
    matthews_corrcoef, cohen_kappa_score,
    top_k_accuracy_score,
)
from sklearn.preprocessing import label_binarize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import csv

# ── CONFIG ────────────────────────────────────────────────────────
DATASET_PATH = "D:\Animals_dataset\Animals_dataset"
SEED         = 42
BATCH_SIZE   = 32
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODELS_CFG = {
    "ResNet50":          ("animal_classifier_resnet50.pth",        "resnet50"),
    "EfficientNet-B0":   ("animal_classifier_efficientnet_b0.pth", "efficientnet_b0"),
    "MobileNetV3-Large": ("animal_classifier_mobilenet_v3.pth",    "mobilenet_v3"),
}

# ── FOLDER → CLEAN CLASS NAME MAP ────────────────────────────────
FOLDER_TO_CLASS = {
    'Peacock':                'Peacock',
    'bonnet_macaque_dataset': 'bonnet_macaque',
    'chital_dataset':         'chital',
    'dataset':                'no_animal',
    'dataset2':               'no_animal',
    'elephant_dataset':       'elephant',
    'pig_dataset':            'pig',
    'porcupine_dataset':      'porcupine',
    'street_dogs_dataset':    'street_dogs',
    'wild_boar_dataset':      'wild_boar',
}

# ── REMAPPED DATASET ──────────────────────────────────────────────
class RemappedDataset(Dataset):
    def __init__(self, root, transform):
        self.base    = datasets.ImageFolder(root, transform=transform)
        raw_classes  = self.base.classes
        clean_names      = [FOLDER_TO_CLASS.get(c, c) for c in raw_classes]
        self.classes     = sorted(set(clean_names))
        clean_to_idx     = {c: i for i, c in enumerate(self.classes)}
        old_to_new = {
            self.base.class_to_idx[raw]: clean_to_idx[FOLDER_TO_CLASS.get(raw, raw)]
            for raw in raw_classes
        }
        self.samples   = [(p, old_to_new[l]) for p, l in self.base.samples]
        self.transform = transform
        self.loader    = self.base.loader

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = self.loader(path)
        if self.transform:
            img = self.transform(img)
        return img, label


# ── MODEL BUILDER ─────────────────────────────────────────────────
def build_model(arch, n_classes, weights_path):
    if arch == "resnet50":
        m = models.resnet50()
        m.fc = nn.Linear(m.fc.in_features, n_classes)

    elif arch == "efficientnet_b0":
        m = models.efficientnet_b0()
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, n_classes)

    elif arch == "mobilenet_v3":
        m = models.mobilenet_v3_large()
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, n_classes)

    else:
        raise ValueError(f"Unknown arch: {arch}")

    m.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    return m.to(DEVICE).eval()


# ── INFERENCE ─────────────────────────────────────────────────────
def run_inference(model, val_loader, n_samples):
    all_labels = []
    all_preds  = []
    all_probs  = []
    t0 = time.time()

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(DEVICE)
            logits = model(inputs)
            probs  = torch.softmax(logits, dim=1).cpu().numpy()
            preds  = probs.argmax(axis=1)
            all_labels.extend(labels.numpy())
            all_preds.extend(preds)
            all_probs.extend(probs)

    elapsed = time.time() - t0
    fps     = n_samples / elapsed
    return (np.array(all_labels),
            np.array(all_preds),
            np.array(all_probs),
            elapsed, fps)


# ── METRICS ───────────────────────────────────────────────────────
def compute_metrics(y_true, y_pred, y_prob, model_name, elapsed, fps):
    acc      = accuracy_score(y_true, y_pred)
    bal_acc  = balanced_accuracy_score(y_true, y_pred)
    top2     = top_k_accuracy_score(y_true, y_prob, k=2)
    top3     = top_k_accuracy_score(y_true, y_prob, k=3)
    prec_mac = precision_score(y_true, y_pred, average="macro",    zero_division=0)
    prec_wt  = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec_mac  = recall_score   (y_true, y_pred, average="macro",    zero_division=0)
    rec_wt   = recall_score   (y_true, y_pred, average="weighted", zero_division=0)
    f1_mac   = f1_score       (y_true, y_pred, average="macro",    zero_division=0)
    f1_wt    = f1_score       (y_true, y_pred, average="weighted", zero_division=0)
    mcc      = matthews_corrcoef(y_true, y_pred)
    kappa    = cohen_kappa_score(y_true, y_pred)

    try:
        auc_mac = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
        auc_wt  = roc_auc_score(y_true, y_prob, multi_class="ovr", average="weighted")
    except Exception:
        auc_mac = auc_wt = float("nan")

    return {
        "Model":              model_name,
        "Accuracy":           round(acc,      4),
        "Balanced Accuracy":  round(bal_acc,  4),
        "Top-2 Accuracy":     round(top2,     4),
        "Top-3 Accuracy":     round(top3,     4),
        "Precision (macro)":  round(prec_mac, 4),
        "Precision (wt)":     round(prec_wt,  4),
        "Recall (macro)":     round(rec_mac,  4),
        "Recall (wt)":        round(rec_wt,   4),
        "F1 (macro)":         round(f1_mac,   4),
        "F1 (weighted)":      round(f1_wt,    4),
        "MCC":                round(mcc,      4),
        "Cohen Kappa":        round(kappa,    4),
        "ROC-AUC (macro)":    round(auc_mac,  4),
        "ROC-AUC (wt)":       round(auc_wt,   4),
        "Inference time (s)": round(elapsed,  2),
        "FPS":                round(fps,      1),
    }


# ── PLOTS ─────────────────────────────────────────────────────────
def plot_confusion_matrix(y_true, y_pred, class_names, model_name, filename):
    cm     = confusion_matrix(y_true, y_pred)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    fig, axes = plt.subplots(1, 2, figsize=(22, 9))
    fig.suptitle(f"Confusion Matrix — {model_name}", fontsize=15, fontweight="bold")

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names,
                linewidths=0.5, ax=axes[0])
    axes[0].set_title("Raw Counts", fontsize=12)
    axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("Actual")
    axes[0].tick_params(axis="x", rotation=35)

    annot = np.array([[f"{v:.1f}%" for v in row] for row in cm_pct])
    sns.heatmap(cm_pct, annot=annot, fmt="", cmap="YlOrRd",
                xticklabels=class_names, yticklabels=class_names,
                linewidths=0.5, vmin=0, vmax=100, ax=axes[1])
    axes[1].set_title("Row-Normalised (%)", fontsize=12)
    axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("Actual")
    axes[1].tick_params(axis="x", rotation=35)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {filename}")


def plot_per_class(y_true, y_pred, class_names, model_name, filename):
    report = classification_report(y_true, y_pred, target_names=class_names,
                                   output_dict=True, zero_division=0)
    keys   = ["precision", "recall", "f1-score", "support"]
    rows   = [[c] + [f"{report[c][k]:.3f}" if k != "support" else str(int(report[c][k]))
               for k in keys] for c in class_names]

    fig, ax = plt.subplots(figsize=(14, 5))
    fig.suptitle(f"Per-Class Metrics — {model_name}", fontsize=13, fontweight="bold")
    ax.axis("off")

    col_labels = ["Class", "Precision", "Recall", "F1-Score", "Support"]
    tbl = ax.table(cellText=rows, colLabels=col_labels, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1, 1.6)
    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor("#2c3e50")
        tbl[0, j].set_text_props(color="white", fontweight="bold")
    for i in range(1, len(rows) + 1):
        for j in range(len(col_labels)):
            tbl[i, j].set_facecolor("#ecf0f1" if i % 2 == 0 else "white")

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {filename}")


def plot_roc_curves(results_dict, class_names, filename):
    n_classes = len(class_names)
    ncols     = 3
    nrows     = (n_classes + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 5 * nrows))
    fig.suptitle("ROC Curves per Class (One-vs-Rest) — All Models",
                 fontsize=14, fontweight="bold")
    axes  = axes.flatten()
    cols  = ["#e74c3c", "#2980b9", "#27ae60"]
    y_bin = label_binarize(
        list(results_dict.values())[0]["y_true"],
        classes=list(range(n_classes))
    )

    for ci, cname in enumerate(class_names):
        ax = axes[ci]
        for mi, (mname, data) in enumerate(results_dict.items()):
            fpr, tpr, _ = roc_curve(y_bin[:, ci], data["y_prob"][:, ci])
            ax.plot(fpr, tpr, color=cols[mi], lw=2,
                    label=f"{mname} (AUC={auc(fpr,tpr):.3f})")
        ax.plot([0,1],[0,1],"k--", lw=1)
        ax.set_xlim([0,1]); ax.set_ylim([0,1.02])
        ax.set_title(cname, fontsize=11, fontweight="bold")
        ax.set_xlabel("FPR", fontsize=9); ax.set_ylabel("TPR", fontsize=9)
        ax.legend(fontsize=8, loc="lower right"); ax.grid(alpha=0.3)

    for ci in range(len(class_names), len(axes)):
        axes[ci].set_visible(False)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {filename}")


def plot_comparison(all_metrics, filename):
    keys   = ["Accuracy","Balanced Accuracy","F1 (macro)","F1 (weighted)",
              "Precision (macro)","Recall (macro)","ROC-AUC (macro)","MCC","Cohen Kappa"]
    names  = [m["Model"] for m in all_metrics]
    cols   = ["#e74c3c","#2980b9","#27ae60"]
    fig, axes = plt.subplots(3, 3, figsize=(18, 13))
    fig.suptitle("Model Comparison — All Metrics", fontsize=15, fontweight="bold")
    axes = axes.flatten()

    for i, key in enumerate(keys):
        ax   = axes[i]
        vals = [m[key] for m in all_metrics]
        bars = ax.bar(names, vals, color=cols, edgecolor="white", linewidth=0.8)
        ax.set_title(key, fontsize=11, fontweight="bold")
        ax.set_ylim(max(0, min(vals)-0.05), min(1.05, max(vals)+0.08))
        ax.set_ylabel("Score", fontsize=9)
        ax.tick_params(axis="x", rotation=15, labelsize=8)
        ax.grid(axis="y", alpha=0.3)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                    f"{val:.4f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {filename}")


def plot_metrics_heatmap(all_metrics, filename):
    keys  = ["Accuracy","Balanced Accuracy","Top-2 Accuracy","Top-3 Accuracy",
             "Precision (macro)","Recall (macro)","F1 (macro)","F1 (weighted)",
             "ROC-AUC (macro)","MCC","Cohen Kappa"]
    data  = np.array([[m[k] for k in keys] for m in all_metrics])
    names = [m["Model"] for m in all_metrics]

    fig, ax = plt.subplots(figsize=(17, 4))
    sns.heatmap(data, annot=True, fmt=".4f", cmap="RdYlGn",
                xticklabels=keys, yticklabels=names,
                vmin=0, vmax=1, linewidths=0.5, ax=ax, annot_kws={"size": 9})
    ax.set_title("Metrics Heatmap — All Models", fontsize=13, fontweight="bold")
    ax.tick_params(axis="x", rotation=35, labelsize=9)
    ax.tick_params(axis="y", rotation=0,  labelsize=10)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {filename}")


def plot_speed(all_metrics, filename):
    names = [m["Model"]    for m in all_metrics]
    fps   = [m["FPS"]      for m in all_metrics]
    acc   = [m["Accuracy"] for m in all_metrics]
    cols  = ["#e74c3c","#2980b9","#27ae60"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Speed vs Accuracy Tradeoff", fontsize=13, fontweight="bold")

    bars = ax1.bar(names, fps, color=cols, edgecolor="white")
    ax1.set_title("Inference Throughput (img/sec)"); ax1.set_ylabel("FPS")
    ax1.grid(axis="y", alpha=0.3)
    for b, v in zip(bars, fps):
        ax1.text(b.get_x()+b.get_width()/2, b.get_height()+0.3,
                 f"{v:.0f}", ha="center", fontsize=10, fontweight="bold")

    ax2.scatter(fps, acc, c=cols, s=220, zorder=5, edgecolors="black", linewidths=1)
    for n, f, a in zip(names, fps, acc):
        ax2.annotate(n, (f, a), textcoords="offset points", xytext=(8, 4), fontsize=9)
    ax2.set_xlabel("Throughput (FPS)"); ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy vs Speed"); ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {filename}")


# ── MAIN ──────────────────────────────────────────────────────────
if __name__ == '__main__':

    # Build val dataset with clean class names
    val_tf_compose = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    full_ds    = RemappedDataset(DATASET_PATH, val_tf_compose)
    class_names = full_ds.classes
    n_classes   = len(class_names)
    print(f"Classes ({n_classes}): {class_names}")

    total      = len(full_ds)
    train_size = int(0.8 * total)
    val_size   = total - train_size

    _, val_subset = torch.utils.data.random_split(
        full_ds, [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED)
    )

    # num_workers=0 required on Windows
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=0)

    print(f"Validation set: {val_size} images | Device: {DEVICE}\n")

    all_metrics  = []
    results_dict = {}
    report_lines = []

    for model_name, (weights_path, arch) in MODELS_CFG.items():
        print(f"\n{'='*60}")
        print(f"  Evaluating: {model_name}")
        print(f"{'='*60}")

        if not os.path.exists(weights_path):
            print(f"  SKIPPED — {weights_path} not found.")
            continue

        model = build_model(arch, n_classes, weights_path)
        y_true, y_pred, y_prob, elapsed, fps = run_inference(model, val_loader, val_size)

        metrics = compute_metrics(y_true, y_pred, y_prob, model_name, elapsed, fps)
        all_metrics.append(metrics)
        results_dict[model_name] = {"y_true": y_true, "y_pred": y_pred, "y_prob": y_prob}

        report = classification_report(y_true, y_pred,
                                       target_names=class_names, zero_division=0)
        header = f"\n--- {model_name} ---\n"
        print(header + report)
        report_lines.append(header + report)
        for k, v in metrics.items():
            line = f"  {k:<25} {v}"
            print(line); report_lines.append(line)

        slug = arch
        plot_confusion_matrix(y_true, y_pred, class_names, model_name,
                              f"confusion_matrix_{slug}.png")
        plot_per_class(y_true, y_pred, class_names, model_name,
                       f"per_class_metrics_{slug}.png")

    # Cross-model plots
    if len(all_metrics) > 1:
        print(f"\n{'='*60}")
        print("  Generating comparison plots ...")
        print(f"{'='*60}")
        plot_roc_curves(results_dict, class_names, "roc_curves.png")
        plot_comparison(all_metrics,               "comparison_bar_chart.png")
        plot_metrics_heatmap(all_metrics,          "metrics_heatmap.png")
        plot_speed(all_metrics,                    "speed_accuracy.png")

    # CSV
    if all_metrics:
        with open("metrics_comparison.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=all_metrics[0].keys())
            w.writeheader(); w.writerows(all_metrics)
        print("\n  Saved: metrics_comparison.csv")

    # Text report
    with open("metrics_report.txt", "w") as f:
        f.write("MODEL COMPARISON REPORT\n" + "="*60 + "\n\n")
        f.writelines(line + "\n" for line in report_lines)
        if all_metrics:
            f.write("\n\nSUMMARY TABLE\n" + "="*60 + "\n")
            hdr = f"{'Metric':<26}" + "".join(f"{m['Model']:>22}" for m in all_metrics)
            f.write(hdr + "\n" + "-"*len(hdr) + "\n")
            for key in all_metrics[0]:
                if key == "Model": continue
                row = f"{key:<26}" + "".join(f"{str(m[key]):>22}" for m in all_metrics)
                f.write(row + "\n")
    print("  Saved: metrics_report.txt")

    # Final console table
    if all_metrics:
        print(f"\n{'='*60}  FINAL RESULTS")
        key_metrics = ["Accuracy","Balanced Accuracy","F1 (macro)",
                       "ROC-AUC (macro)","MCC","Cohen Kappa","FPS"]
        hdr = f"  {'Metric':<26}" + "".join(f"{m['Model']:>22}" for m in all_metrics)
        print(hdr); print("  " + "-"*(len(hdr)-2))
        for key in key_metrics:
            row = f"  {key:<26}" + "".join(f"{str(m[key]):>22}" for m in all_metrics)
            print(row)
        print(f"\n  Best Accuracy  : {max(all_metrics, key=lambda x: x['Accuracy'])['Model']}")
        print(f"  Best F1 (macro): {max(all_metrics, key=lambda x: x['F1 (macro)'])['Model']}")
        print(f"  Fastest        : {max(all_metrics, key=lambda x: x['FPS'])['Model']}")