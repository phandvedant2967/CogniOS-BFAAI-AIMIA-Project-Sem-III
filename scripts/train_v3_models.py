"""Stage 4D: Train & Fairly Compare XGBoost and CNN-GRU on V3 Dataset.

Dataset: data/v3/processed/ (5,000 cases)
Splits:
  - train (3,500 patients): for fitting models
  - val (750 patients): for model selection and early stopping
  - final_test (750 patients): held-out, evaluated strictly ONCE after final selection

Models:
  1. XGBoost Baseline: Gradient boosted trees on engineered static tabular features
  2. Longitudinal CNN-GRU: 1D CNN + GRU on chronological observation sequences

Deliverables:
  - Checkpoint: models/v3_cnn_gru.pt
  - Checkpoint: models/v3_xgboost.joblib
  - Metrics: results/v3_metrics.json
  - Plots: results/v3_learning_curves.png, results/v3_confusion_matrices.png, results/v3_calibration.png
  - Report: results/STAGE4D_REPORT.md
"""

import os
import sys
import json
import random
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.calibration import calibration_curve
import xgboost as xgb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from ml.data import TARGET_CLASSES, get_feature_columns
from ml.preprocessing import PreprocessorPipeline
from ml.sequence_interface import SequenceFeatureBuilder, SequenceDataset
from ml.evaluation import evaluate_predictions
from ml.utils import set_seed, save_json, MODELS_DIR, RESULTS_DIR

PROC_V3 = os.path.join(ROOT, "data", "v3", "processed")


def set_all_seeds(seed=42):
    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class V3LongitudinalCNNGRU(nn.Module):
    """Longitudinal 1D CNN + GRU model for V3 dataset."""

    def __init__(
        self,
        input_dim=41,
        conv_dim=64,
        gru_dim=64,
        num_classes=4,
        dropout=0.30,
        kernel_size=3
    ):
        super().__init__()
        self.input_dim = input_dim
        self.conv_dim = conv_dim
        self.gru_dim = gru_dim
        self.num_classes = num_classes

        # 1D CNN layers over temporal dimension
        self.conv1 = nn.Conv1d(input_dim, conv_dim, kernel_size=kernel_size, padding=kernel_size // 2)
        self.bn1 = nn.BatchNorm1d(conv_dim)
        self.conv2 = nn.Conv1d(conv_dim, conv_dim, kernel_size=kernel_size, padding=kernel_size // 2)
        self.bn2 = nn.BatchNorm1d(conv_dim)
        self.conv_drop = nn.Dropout(p=dropout)

        # GRU layer
        self.gru = nn.GRU(
            input_size=conv_dim,
            hidden_size=gru_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )

        # Pooling & Classification
        pooling_dim = gru_dim * 2
        self.classifier = nn.Sequential(
            nn.Linear(pooling_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x, lengths):
        # x: [B, T, D] -> Conv1D expects [B, D, T]
        batch_size, seq_len, _ = x.shape
        x_trans = x.transpose(1, 2)

        c1 = F.relu(self.bn1(self.conv1(x_trans)))
        c2 = F.relu(self.bn2(self.conv2(c1)))
        c_out = self.conv_drop(c2).transpose(1, 2)  # [B, T, conv_dim]

        # Dynamic mask for padding
        mask = torch.arange(seq_len, device=x.device).unsqueeze(0) < lengths.unsqueeze(1)
        c_out = c_out * mask.unsqueeze(-1).float()

        gru_out, _ = self.gru(c_out)  # [B, T, gru_dim]

        # Gather last valid time step hidden state
        idx = (lengths - 1).view(-1, 1, 1).expand(-1, 1, self.gru_dim)
        idx = torch.clamp(idx, min=0)
        last_hidden = gru_out.gather(1, idx).squeeze(1)  # [B, gru_dim]

        # Mean pooling over valid timesteps
        mask_expanded = mask.unsqueeze(-1).float()
        sum_hidden = (gru_out * mask_expanded).sum(dim=1)
        lengths_expanded = torch.clamp(lengths.unsqueeze(1).float(), min=1.0)
        mean_hidden = sum_hidden / lengths_expanded  # [B, gru_dim]

        pooled = torch.cat([last_hidden, mean_hidden], dim=-1)  # [B, gru_dim * 2]
        logits = self.classifier(pooled)
        return logits


def train_xgboost(X_tr, y_tr, X_va, y_va):
    print("\n--- Training Baseline XGBoost on V3 ---")
    cls_to_idx = {c: i for i, c in enumerate(TARGET_CLASSES)}
    y_tr_idx = np.array([cls_to_idx[c] for c in y_tr])
    y_va_idx = np.array([cls_to_idx[c] for c in y_va])

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=2,
        random_state=42,
        eval_metric="mlogloss",
        early_stopping_rounds=25,
        n_jobs=-1
    )

    model.fit(
        X_tr, y_tr_idx,
        eval_set=[(X_tr, y_tr_idx), (X_va, y_va_idx)],
        verbose=50
    )

    val_preds_prob = model.predict_proba(X_va)
    val_preds_idx = np.argmax(val_preds_prob, axis=1)
    val_preds = [TARGET_CLASSES[i] for i in val_preds_idx]

    metrics_va = evaluate_predictions(y_va.tolist(), val_preds, val_preds_prob)
    print(f"XGBoost Val Macro-F1: {metrics_va['macro_f1']:.4f} | Val Accuracy: {metrics_va['accuracy']:.4f}")
    return model, metrics_va


def train_cnn_gru(train_loader, val_loader, input_dim, epochs=40, lr=1e-3, weight_decay=1e-4):
    print(f"\n--- Training CNN-GRU on V3 (Epochs={epochs}, LR={lr}, Weight Decay={weight_decay}) ---")
    model = V3LongitudinalCNNGRU(
        input_dim=input_dim,
        conv_dim=64,
        gru_dim=64,
        num_classes=4,
        dropout=0.30
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)

    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "train_macro_f1": [],
        "val_macro_f1": [],
        "val_accuracy": []
    }

    best_val_f1 = -1.0
    best_checkpoint_path = os.path.join(MODELS_DIR, "v3_cnn_gru.pt")
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        # 1. Training
        model.train()
        train_loss_total = 0.0
        train_preds, train_targets = [], []

        for batch in train_loader:
            seqs = batch["sequence"]
            lens = batch["length"]
            targs = batch["target"]

            optimizer.zero_grad()
            logits = model(seqs, lens)
            loss = criterion(logits, targs)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

            train_loss_total += loss.item() * len(targs)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            train_preds.extend(preds)
            train_targets.extend(targs.cpu().numpy())

        train_loss = train_loss_total / len(train_targets)
        train_y_true = [TARGET_CLASSES[i] for i in train_targets]
        train_y_pred = [TARGET_CLASSES[i] for i in train_preds]
        tr_m = evaluate_predictions(train_y_true, train_y_pred)

        # 2. Validation
        model.eval()
        val_loss_total = 0.0
        val_preds, val_targets = [], []
        val_probs_all = []

        with torch.no_grad():
            for batch in val_loader:
                seqs = batch["sequence"]
                lens = batch["length"]
                targs = batch["target"]

                logits = model(seqs, lens)
                loss = criterion(logits, targs)

                val_loss_total += loss.item() * len(targs)
                probs = F.softmax(logits, dim=1).cpu().numpy()
                preds = np.argmax(probs, axis=1)

                val_preds.extend(preds)
                val_targets.extend(targs.cpu().numpy())
                val_probs_all.append(probs)

        val_loss = val_loss_total / len(val_targets)
        val_probs = np.vstack(val_probs_all)
        val_y_true = [TARGET_CLASSES[i] for i in val_targets]
        val_y_pred = [TARGET_CLASSES[i] for i in val_preds]
        va_m = evaluate_predictions(val_y_true, val_y_pred, val_probs)

        scheduler.step(va_m["macro_f1"])

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_macro_f1"].append(tr_m["macro_f1"])
        history["val_macro_f1"].append(va_m["macro_f1"])
        history["val_accuracy"].append(va_m["accuracy"])

        if va_m["macro_f1"] > best_val_f1:
            best_val_f1 = va_m["macro_f1"]
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_macro_f1": best_val_f1,
                "val_metrics": va_m,
                "input_dim": input_dim
            }, best_checkpoint_path)

        if epoch % 5 == 0 or epoch == epochs:
            print(f"Epoch {epoch:2d}/{epochs:2d} | TrLoss: {train_loss:.4f} TrF1: {tr_m['macro_f1']:.4f} | VaLoss: {val_loss:.4f} VaF1: {va_m['macro_f1']:.4f} VaAcc: {va_m['accuracy']:.4f} (Best: {best_val_f1:.4f} @ Ep {best_epoch})")

    # Load best model
    checkpoint = torch.load(best_checkpoint_path, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded best checkpoint from epoch {best_epoch} with Val Macro-F1: {best_val_f1:.4f}")
    return model, checkpoint["val_metrics"], history


def evaluate_final_test_cnn_gru(model, test_loader):
    model.eval()
    test_preds, test_targets = [], []
    test_probs_all = []

    with torch.no_grad():
        for batch in test_loader:
            seqs = batch["sequence"]
            lens = batch["length"]
            targs = batch["target"]

            logits = model(seqs, lens)
            probs = F.softmax(logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            test_preds.extend(preds)
            test_targets.extend(targs.cpu().numpy())
            test_probs_all.append(probs)

    test_probs = np.vstack(test_probs_all)
    y_true = [TARGET_CLASSES[i] for i in test_targets]
    y_pred = [TARGET_CLASSES[i] for i in test_preds]
    return evaluate_predictions(y_true, y_pred, test_probs), y_true, y_pred, test_probs


def evaluate_final_test_xgboost(model, X_test, y_test):
    cls_to_idx = {c: i for i, c in enumerate(TARGET_CLASSES)}
    test_probs = model.predict_proba(X_test)
    preds_idx = np.argmax(test_probs, axis=1)
    y_pred = [TARGET_CLASSES[i] for i in preds_idx]
    y_true = y_test.tolist()
    return evaluate_predictions(y_true, y_pred, test_probs), y_true, y_pred, test_probs


def plot_learning_curves(history, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history["epoch"], history["train_loss"], label="Train Loss", color="royalblue", lw=2)
    axes[0].plot(history["epoch"], history["val_loss"], label="Val Loss", color="darkorange", lw=2)
    axes[0].set_xlabel("Epoch", fontsize=12)
    axes[0].set_ylabel("Cross-Entropy Loss", fontsize=12)
    axes[0].set_title("V3 CNN-GRU Loss Learning Curve", fontsize=14, fontweight="bold")
    axes[0].legend(fontsize=11)
    axes[0].grid(True, linestyle="--", alpha=0.6)

    axes[1].plot(history["epoch"], history["train_macro_f1"], label="Train Macro-F1", color="royalblue", lw=2)
    axes[1].plot(history["epoch"], history["val_macro_f1"], label="Val Macro-F1", color="darkorange", lw=2)
    axes[1].set_xlabel("Epoch", fontsize=12)
    axes[1].set_ylabel("Macro-F1 Score", fontsize=12)
    axes[1].set_title("V3 CNN-GRU Macro-F1 Curve", fontsize=14, fontweight="bold")
    axes[1].legend(fontsize=11)
    axes[1].grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved learning curves plot -> {out_path}")


def plot_confusion_matrices(cm_xgb, cm_cnn, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    classes_short = ["Med-Assoc", "Progression", "Other Cause", "Uncertain"]

    # XGBoost CM
    cm_xgb_mat = np.array([[cm_xgb[r][c] for c in TARGET_CLASSES] for r in TARGET_CLASSES])
    sns.heatmap(cm_xgb_mat, annot=True, fmt="d", cmap="Blues", ax=axes[0],
                xticklabels=classes_short, yticklabels=classes_short, cbar=False)
    axes[0].set_title("XGBoost Confusion Matrix (V3 Final-Test)", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Predicted Label", fontsize=11)
    axes[0].set_ylabel("True Label", fontsize=11)

    # CNN-GRU CM
    cm_cnn_mat = np.array([[cm_cnn[r][c] for c in TARGET_CLASSES] for r in TARGET_CLASSES])
    sns.heatmap(cm_cnn_mat, annot=True, fmt="d", cmap="Greens", ax=axes[1],
                xticklabels=classes_short, yticklabels=classes_short, cbar=False)
    axes[1].set_title("Longitudinal CNN-GRU Confusion Matrix (V3 Final-Test)", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Predicted Label", fontsize=11)
    axes[1].set_ylabel("True Label", fontsize=11)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved confusion matrices plot -> {out_path}")


def plot_calibration_curves(y_true, xgb_probs, cnn_probs, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    for idx, cls in enumerate(TARGET_CLASSES):
        y_binary = np.array([1 if y == cls else 0 for y in y_true])

        prob_true_xgb, prob_pred_xgb = calibration_curve(y_binary, xgb_probs[:, idx], n_bins=8, strategy="uniform")
        prob_true_cnn, prob_pred_cnn = calibration_curve(y_binary, cnn_probs[:, idx], n_bins=8, strategy="uniform")

        axes[idx].plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
        axes[idx].plot(prob_pred_xgb, prob_true_xgb, "s-", color="royalblue", label="XGBoost")
        axes[idx].plot(prob_pred_cnn, prob_true_cnn, "o-", color="darkgreen", label="CNN-GRU")

        axes[idx].set_title(f"Class: {cls}", fontsize=12, fontweight="bold")
        axes[idx].set_xlabel("Mean Predicted Probability", fontsize=10)
        axes[idx].set_ylabel("Fraction of Positives", fontsize=10)
        axes[idx].legend(loc="lower right")
        axes[idx].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved calibration curve plot -> {out_path}")


def main():
    set_all_seeds(42)
    print("==================================================================")
    print("STAGE 4D: Training and Fair Evaluation on Diverse V3 Cohort")
    print("==================================================================")

    # 1. Load V3 Tabular Cases
    cases_path = os.path.join(PROC_V3, "ml_cases.csv")
    df_cases = pd.read_csv(cases_path)
    usable_df = df_cases[df_cases.meta_usable_case == 1].copy()
    print(f"Loaded {len(usable_df)} usable cases from {cases_path}")

    # Splits
    train_df = usable_df[usable_df.split == "train"].copy()
    val_df = usable_df[usable_df.split == "val"].copy()
    test_df = usable_df[usable_df.split == "final_test"].copy()

    print(f"Train Cases: {len(train_df)} | Val Cases: {len(val_df)} | Final Test Cases: {len(test_df)}")

    feat_cols = get_feature_columns(train_df)
    X_train_raw = train_df[feat_cols]
    y_train = train_df["target"]
    X_val_raw = val_df[feat_cols]
    y_val = val_df["target"]
    X_test_raw = test_df[feat_cols]
    y_test = test_df["target"]

    # Preprocessor (fit strictly on train)
    print("\nFitting tabular preprocessor strictly on V3 train split...")
    prep = PreprocessorPipeline()
    X_train_proc = prep.fit_transform(X_train_raw)
    X_val_proc = prep.transform(X_val_raw)
    X_test_proc = prep.transform(X_test_raw)
    joblib.dump(prep, os.path.join(MODELS_DIR, "v3_preprocessor.joblib"))

    # 2. Train Baseline XGBoost
    xgb_model, xgb_val_metrics = train_xgboost(X_train_proc, y_train, X_val_proc, y_val)
    joblib.dump(xgb_model, os.path.join(MODELS_DIR, "v3_xgboost.joblib"))

    # 3. Build Longitudinal Sequence Datasets
    print("\nExtracting longitudinal sequence matrices using SequenceFeatureBuilder...")
    builder = SequenceFeatureBuilder(proc_dir=PROC_V3, max_seq_len=14)
    raw_records = builder.build_raw_patient_sequences()
    datasets = builder.fit_and_transform(raw_records)

    train_ds = datasets["train"]
    val_ds = datasets["val"]
    test_ds = datasets["final_test"]

    print(f"Sequence Datasets built: Train={len(train_ds)}, Val={len(val_ds)}, Final-Test={len(test_ds)}")
    input_dim = len(builder.feature_names)
    print(f"Sequence Feature Dimension: {input_dim}")

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    # 4. Train CNN-GRU on V3
    cnn_model, cnn_val_metrics, history = train_cnn_gru(
        train_loader, val_loader, input_dim=input_dim, epochs=35, lr=1e-3, weight_decay=1e-4
    )

    # Plot learning curves
    plot_learning_curves(history, os.path.join(RESULTS_DIR, "v3_learning_curves.png"))

    # 5. Model Selection Review
    print("\n=======================================================")
    print("VALIDATION COMPARISON (Used for Model Selection)")
    print("=======================================================")
    print(f"XGBoost Validation:  Macro-F1 = {xgb_val_metrics['macro_f1']:.4f} | Accuracy = {xgb_val_metrics['accuracy']:.4f}")
    print(f"CNN-GRU Validation:  Macro-F1 = {cnn_val_metrics['macro_f1']:.4f} | Accuracy = {cnn_val_metrics['accuracy']:.4f}")

    # 6. Final Evaluation on Fresh Held-Out Cohort (STRICTLY ONCE)
    print("\n==================================================================")
    print("FINAL HELD-OUT TEST EVALUATION (Fresh Cohort: 750 Patients)")
    print("==================================================================")

    xgb_test_metrics, y_test_true, xgb_test_pred, xgb_test_probs = evaluate_final_test_xgboost(xgb_model, X_test_proc, y_test)
    cnn_test_metrics, _, cnn_test_pred, cnn_test_probs = evaluate_final_test_cnn_gru(cnn_model, test_loader)

    print("\n--- FINAL TEST RESULTS: XGBoost Baseline ---")
    print(f"Accuracy:        {xgb_test_metrics['accuracy']:.4f}")
    print(f"Macro-F1:        {xgb_test_metrics['macro_f1']:.4f}")
    print(f"Weighted-F1:     {xgb_test_metrics['weighted_f1']:.4f}")
    print(f"Macro-Precision: {xgb_test_metrics['macro_precision']:.4f}")
    print(f"Macro-Recall:    {xgb_test_metrics['macro_recall']:.4f}")
    print(f"Macro ROC-AUC:   {xgb_test_metrics.get('macro_roc_auc', 0.0):.4f}")
    for c in TARGET_CLASSES:
        pc = xgb_test_metrics['per_class'][c]
        print(f"  [{c}]: F1={pc['f1']:.4f}, Recall={pc['recall']:.4f}, Precision={pc['precision']:.4f}")

    print("\n--- FINAL TEST RESULTS: Longitudinal CNN-GRU ---")
    print(f"Accuracy:        {cnn_test_metrics['accuracy']:.4f}")
    print(f"Macro-F1:        {cnn_test_metrics['macro_f1']:.4f}")
    print(f"Weighted-F1:     {cnn_test_metrics['weighted_f1']:.4f}")
    print(f"Macro-Precision: {cnn_test_metrics['macro_precision']:.4f}")
    print(f"Macro-Recall:    {cnn_test_metrics['macro_recall']:.4f}")
    print(f"Macro ROC-AUC:   {cnn_test_metrics.get('macro_roc_auc', 0.0):.4f}")
    for c in TARGET_CLASSES:
        pc = cnn_test_metrics['per_class'][c]
        print(f"  [{c}]: F1={pc['f1']:.4f}, Recall={pc['recall']:.4f}, Precision={pc['precision']:.4f}")

    # Plot Confusion Matrices & Calibration
    plot_confusion_matrices(
        xgb_test_metrics["confusion_matrix"],
        cnn_test_metrics["confusion_matrix"],
        os.path.join(RESULTS_DIR, "v3_confusion_matrices.png")
    )

    plot_calibration_curves(
        y_test_true,
        xgb_test_probs,
        cnn_test_probs,
        os.path.join(RESULTS_DIR, "v3_calibration.png")
    )

    # 7. Save JSON Metrics
    v3_results = {
        "dataset": "V3 Diverse Realistic Cohort (5,000 cases)",
        "train_usable": len(train_df),
        "val_usable": len(val_df),
        "test_usable": len(test_df),
        "models": {
            "xgboost": {
                "validation": xgb_val_metrics,
                "final_test": xgb_test_metrics
            },
            "cnn_gru": {
                "validation": cnn_val_metrics,
                "final_test": cnn_test_metrics,
                "history": history
            }
        }
    }
    save_json(v3_results, os.path.join(RESULTS_DIR, "v3_metrics.json"))
    print("\nSaved V3 metrics -> results/v3_metrics.json")


if __name__ == "__main__":
    main()
