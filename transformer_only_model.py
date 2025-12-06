"""
Transformer-Only Model for Objective Prediction
Uses only temporal sequence features 
Predicts team-specific dragon/baron kills in the next 3 minutes.
"""

import argparse
import json
import math
import os
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Tuple, Dict, List
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

#wandb setup 
WANDB_AVAILABLE = False
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    pass

USE_WANDB = os.getenv("USE_WANDB", "1").lower() not in {"0", "false", "off", "no"}
WANDB_MODE = os.getenv("WANDB_MODE", "offline")
WANDB_PROJECT = os.getenv("WANDB_PROJECT", "transformer-only-model")
WANDB_ENTITY = os.getenv("WANDB_ENTITY")


def init_wandb_run(name: str, job_type: str, config: dict | None = None, tags: list[str] | None = None):
    if not (USE_WANDB and WANDB_AVAILABLE):
        return None
    try:
        return wandb.init(
            project=WANDB_PROJECT,
            entity=WANDB_ENTITY,
            name=name,
            job_type=job_type,
            tags=tags,
            mode=WANDB_MODE,
            config=config or {},
            reinit=True,
        )
    except Exception as exc:
        print(f"[WARN] wandb init failed: {exc}. Continuing without logging.")
        return None

#initialize constants
DATA_ROOT = Path(os.getenv("DATA_ROOT", "/project/project_465002423/Deep-Learning-Project-2025/combined_data"))
#support both single file and directory 
TEAM_SEQUENCE_FEATURES_PATH = Path(os.getenv("TEAM_SEQUENCE_FEATURES_PATH", str(DATA_ROOT / "team_sequence_dataset")))
if not TEAM_SEQUENCE_FEATURES_PATH.is_absolute():
    TEAM_SEQUENCE_FEATURES_PATH = DATA_ROOT / TEAM_SEQUENCE_FEATURES_PATH
TEAM_SEQUENCE_METADATA_PATH = DATA_ROOT / "team_sequence_metadata.json"

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

BATCH_SIZE = 64
MAX_SEQ_LEN = 64
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1
RANDOM_SEED = 42
MAX_EPOCHS = 40
EARLY_STOPPING_PATIENCE = 7
WEIGHT_DECAY = 1e-4

_slurm_cpus = os.getenv("SLURM_CPUS_PER_TASK")
if _slurm_cpus:
    NUM_WORKERS = max(1, int(_slurm_cpus) - 2)
else:
    NUM_WORKERS = 4

OUTPUT_DIR = DATA_ROOT / "transformer_only_model"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class TrainConfig:
    d_model: int = 128
    n_layers: int = 2
    d_ff_multiplier: float = 3.0
    transformer_dropout: float = 0.1
    max_history_minutes: str | int = "all"
    batch_size: int = BATCH_SIZE
    lr: float = 1e-3
    weight_decay: float = WEIGHT_DECAY
    max_epochs: int = MAX_EPOCHS
    patience: int = EARLY_STOPPING_PATIENCE
    label_smoothing: float = 0.0
    seed: int = RANDOM_SEED

class TransformerEncoder(nn.Module):
    """
    transformer encoder for temporal team features
    """
    def __init__(
        self,
        feature_dim: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        max_seq_len: int = 64,
        dim_feedforward: int | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(feature_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True,
            dropout=dropout,
            dim_feedforward=dim_feedforward or int(4 * d_model),
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        
        #causal mask
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool), diagonal=1)
        )

    def forward(self, x: torch.Tensor, src_key_padding_mask: torch.Tensor | None = None):
        h = self.input_proj(x)
        seq_len = x.size(1)
        causal_mask = self.causal_mask[:seq_len, :seq_len]
        out = self.encoder(h, mask=causal_mask, src_key_padding_mask=src_key_padding_mask)
        out = self.dropout(out)
        return out  #(B, T, d_model)


class TransformerOnlyModel(nn.Module):
    """
    transformer-only model (no spatial data)
    """
    def __init__(
        self,
        sequence_feature_dim: int,
        d_model: int = 128,
        n_layers: int = 2,
        d_ff_multiplier: float = 3.0,
        transformer_dropout: float = 0.1,
        hidden_dim: int = 256,
        hidden_dropout: float = 0.2,
        max_seq_len: int = 64,
        num_labels: int = 2,
    ):
        super().__init__()
        
        #transformer branch
        self.transformer = TransformerEncoder(
            feature_dim=sequence_feature_dim,
            d_model=d_model,
            nhead=4,
            num_layers=n_layers,
            max_seq_len=max_seq_len,
            dim_feedforward=int(d_model * d_ff_multiplier),
            dropout=transformer_dropout,
        )
        
        #prediction heads 
        assert num_labels == 6, "Expected 6 labels: 3 horizons × 2 objectives"
        self.shared_mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(hidden_dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(hidden_dropout),
        )
        
        # Separate heads for each horizon
        self.head_1min = nn.Linear(hidden_dim // 2, 2)  # dragon_1min, baron_1min
        self.head_2min = nn.Linear(hidden_dim // 2, 2)  # dragon_2min, baron_2min
        self.head_3min = nn.Linear(hidden_dim // 2, 2)  # dragon_3min, baron_3min
        
        self.max_seq_len = max_seq_len
        self.num_labels = num_labels

    def forward(
        self,
        sequence_features: torch.Tensor,
        src_key_padding_mask: torch.Tensor | None = None,
        valid_lengths: torch.Tensor = None,
    ):
        """
        Many-to-One forward pass.
        """
        B, T, _ = sequence_features.shape
        device = sequence_features.device
        
        #process sequence
        transformer_out = self.transformer(sequence_features, src_key_padding_mask)  # (B, T, d_model)
        
        #extract embedding at last valid timestep
        if valid_lengths is None:
            valid_lengths = torch.full((B,), T, dtype=torch.long, device=device)
        
        last_indices = (valid_lengths - 1).clamp(min=0)  # (B,)
        batch_indices = torch.arange(B, device=device)
        transformer_last = transformer_out[batch_indices, last_indices]  # (B, d_model)
        
        #shared feature extraction
        shared_features = self.shared_mlp(transformer_last)  # (B, hidden_dim // 2)
        
        #separate heads for each horizon
        logits_1min = self.head_1min(shared_features)  # (B, 2)
        logits_2min = self.head_2min(shared_features)  # (B, 2)
        logits_3min = self.head_3min(shared_features)  # (B, 2)
        
        #concatenate in label order: [dragon_1min, dragon_2min, dragon_3min, baron_1min, baron_2min, baron_3min]
        logits = torch.cat([
            logits_1min[:, 0:1],  # dragon_1min
            logits_2min[:, 0:1],  # dragon_2min
            logits_3min[:, 0:1],  # dragon_3min
            logits_1min[:, 1:2],  # baron_1min
            logits_2min[:, 1:2],  # baron_2min
            logits_3min[:, 1:2],  # baron_3min
        ], dim=-1)  # (B, 6)
        
        return logits

class TransformerOnlyDataset(Dataset):
    """
    Dataset for transformer only model 
    """
    def __init__(
        self,
        sequence_df: pd.DataFrame,
        feature_cols: List[str],
        label_cols: List[str],
        eligibility_cols: List[str] | None = None,
        max_seq_len: int | None = MAX_SEQ_LEN,
        stride: int = 1,
        min_history: int = 1,
    ):
        grouped = sequence_df.sort_values(["matchId", "teamId", "minute"])
        sequences = []
        eligibility_cols = eligibility_cols or []
        
        for (match_id, team_id), group in grouped.groupby(["matchId", "teamId"], sort=False):
            arr = group[feature_cols].to_numpy(dtype=np.float32)
            labels = group[label_cols].to_numpy(dtype=np.float32)
            minutes = group["minute"].to_numpy()
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            labels = np.nan_to_num(labels, nan=0.0, posinf=0.0, neginf=0.0)
            
            if eligibility_cols:
                elig = group[eligibility_cols].to_numpy(dtype=np.float32)
                elig = np.nan_to_num(elig, nan=0.0, posinf=0.0, neginf=0.0)
            else:
                elig = np.ones_like(labels, dtype=np.float32)
            
            if len(arr) < min_history:
                continue
            
            max_len = max_seq_len or len(arr)
            for end_idx in range(min_history, len(arr) + 1, stride):
                start_idx = max(0, end_idx - max_len)
                seq = arr[start_idx:end_idx]
                lbl = labels[start_idx:end_idx]
                elig_slice = elig[start_idx:end_idx]
                
                sequences.append({
                    "seq": seq,
                    "labels": lbl,
                    "elig": elig_slice,
                })
        
        self.sequences = sequences
        if max_seq_len is None:
            self.max_seq_len = max((len(s["seq"]) for s in sequences), default=0)
        else:
            self.max_seq_len = max_seq_len
        self.feature_cols = feature_cols
        self.feature_dim = len(feature_cols)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict:
        item = self.sequences[idx]
        seq, lbl, elig = item["seq"], item["labels"], item["elig"]
        valid_len = len(seq)
        
        #pad sequence
        pad_len = max(self.max_seq_len - valid_len, 0)
        if pad_len > 0:
            seq = np.pad(seq, ((0, pad_len), (0, 0)), mode="constant")
            lbl = np.pad(lbl, ((0, pad_len), (0, 0)), mode="constant")
            elig = np.pad(elig, ((0, pad_len), (0, 0)), mode="constant")
        elif pad_len < 0:
            seq = seq[:self.max_seq_len]
            lbl = lbl[:self.max_seq_len]
            elig = elig[:self.max_seq_len]
            valid_len = self.max_seq_len
        
        return {
            "sequence_inputs": torch.tensor(seq, dtype=torch.float32),
            "labels": torch.tensor(lbl, dtype=torch.float32),
            "eligibility": torch.tensor(elig, dtype=torch.float32),
            "valid_len": valid_len,
        }

def collate_transformer_only(batch):
    """
    custom collate function for transformer-only dataset
    """
    sequence_inputs = torch.stack([item["sequence_inputs"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    eligibility = torch.stack([item["eligibility"] for item in batch])
    lengths = torch.tensor([item["valid_len"] for item in batch], dtype=torch.long)
    mask = (torch.arange(sequence_inputs.shape[1]).unsqueeze(0) < lengths.unsqueeze(1)).float()
    
    return {
        "sequence_inputs": sequence_inputs,
        "labels": labels,
        "eligibility": eligibility,
        "mask": mask,
        "valid_lengths": lengths,
    }

#training functions
def _label_smooth(targets: torch.Tensor, smoothing: float) -> torch.Tensor:
    if smoothing <= 0:
        return targets
    return targets * (1.0 - smoothing) + 0.5 * smoothing

def compute_loss(logits: torch.Tensor, targets: torch.Tensor, valid_mask: torch.Tensor,
                 pos_weight: torch.Tensor, label_smoothing: float) -> torch.Tensor:
    smoothed_targets = _label_smooth(targets, label_smoothing)
    logits = torch.clamp(logits, min=-100, max=100)
    loss_raw = F.binary_cross_entropy_with_logits(
        logits, smoothed_targets, reduction="none", pos_weight=pos_weight
    )
    loss_raw = torch.where(torch.isnan(loss_raw), torch.zeros_like(loss_raw), loss_raw)
    loss = (loss_raw * valid_mask).sum() / (valid_mask.sum() + 1e-6)
    return loss

def evaluate_model(model: nn.Module, loader: DataLoader, pos_weight: torch.Tensor,
                   num_labels: int, device: torch.device, label_smoothing: float = 0.0) -> Dict:
    model.eval()
    total_loss = 0.0
    total_weight = 0.0
    y_true = [[] for _ in range(num_labels)]
    y_score = [[] for _ in range(num_labels)]
    
    with torch.no_grad():
        for batch in loader:
            seq_inputs = batch["sequence_inputs"].to(device)
            labels_full = batch["labels"].to(device)
            eligibility_full = batch["eligibility"].to(device)
            mask = batch["mask"].to(device)
            valid_lengths = batch["valid_lengths"].to(device)
            
            logits = model(seq_inputs, src_key_padding_mask=(mask == 0), valid_lengths=valid_lengths)
            
            logits = torch.clamp(logits, min=-100, max=100)
            logits = torch.where(torch.isnan(logits), torch.zeros_like(logits), logits)
            
            B = seq_inputs.shape[0]
            last_indices = (valid_lengths - 1).clamp(min=0)
            batch_indices = torch.arange(B, device=device)
            labels = labels_full[batch_indices, last_indices]
            eligibility = eligibility_full[batch_indices, last_indices]
            
            preds = torch.sigmoid(logits) * eligibility
            valid_mask = eligibility
            
            batch_loss = compute_loss(logits, labels, valid_mask, pos_weight, label_smoothing)
            batch_weight = valid_mask.sum().item()
            total_loss += batch_loss.item() * batch_weight
            total_weight += batch_weight
            
            valid_bool = valid_mask.bool()
            for idx in range(num_labels):
                idx_mask = valid_bool[:, idx]
                if idx_mask.any():
                    y_true[idx].extend(labels[:, idx][idx_mask].detach().cpu().numpy())
                    y_score[idx].extend(preds[:, idx][idx_mask].detach().cpu().numpy())
    
    per_label_pr = []
    per_label_precision = []
    per_label_recall = []
    per_label_accuracy = []
    
    for idx in range(num_labels):
        targets = np.array(y_true[idx])
        scores = np.array(y_score[idx])
        valid_mask_np = ~(np.isnan(targets) | np.isnan(scores))
        targets = targets[valid_mask_np]
        scores = scores[valid_mask_np]
        
        if len(targets) == 0 or len(np.unique(targets)) < 2:
            per_label_pr.append(float("nan"))
            per_label_precision.append(float("nan"))
            per_label_recall.append(float("nan"))
            per_label_accuracy.append(float("nan"))
        else:
            per_label_pr.append(float(average_precision_score(targets, scores)))
            
            preds_binary = (scores > 0.5).astype(float)
            tp = ((preds_binary == 1) & (targets == 1)).sum()
            fp = ((preds_binary == 1) & (targets == 0)).sum()
            fn = ((preds_binary == 0) & (targets == 1)).sum()
            tn = ((preds_binary == 0) & (targets == 0)).sum()
            
            precision = tp / (tp + fp + 1e-6)
            recall = tp / (tp + fn + 1e-6)
            accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-6)
            
            per_label_precision.append(float(precision))
            per_label_recall.append(float(recall))
            per_label_accuracy.append(float(accuracy))
    
    macro_pr = np.nanmean(per_label_pr)
    macro_precision = np.nanmean(per_label_precision)
    macro_recall = np.nanmean(per_label_recall)
    macro_accuracy = np.nanmean(per_label_accuracy)
    
    return {
        "loss": total_loss / (total_weight + 1e-6),
        "macro_pr_auc": float(macro_pr),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_accuracy": float(macro_accuracy),
        "per_label_pr_auc": per_label_pr,
        "per_label_precision": per_label_precision,
        "per_label_recall": per_label_recall,
        "per_label_accuracy": per_label_accuracy,
    }

def create_scheduler(optimizer: torch.optim.Optimizer, total_steps: int, warmup_ratio: float = 0.1):
    warmup_steps = max(1, int(total_steps * warmup_ratio))
    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = (current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def resolve_max_seq_len(max_history_minutes: str | int) -> int | None:
    if isinstance(max_history_minutes, int):
        return max_history_minutes
    if max_history_minutes == "all":
        return None
    raise ValueError(f"Invalid max_history_minutes: {max_history_minutes}")

def _compute_pos_weight(df: pd.DataFrame, label_col: str, eligibility_col: str | None = None) -> float:
    if eligibility_col and eligibility_col in df.columns:
        elig = df[eligibility_col].astype(np.float32)
    else:
        elig = np.ones(len(df), dtype=np.float32)
    labels = df[label_col].astype(np.float32)
    pos = float((labels * elig).sum())
    neg = float(((1.0 - labels) * elig).sum())
    if pos <= 0.0:
        return 1.0
    return max(1.0, (neg + 1e-6) / (pos + 1e-6))

def train_single_config(
    config: TrainConfig,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    label_cols: List[str],
    eligibility_cols: List[str],
    pos_weight: torch.Tensor,
):
    set_global_seed(config.seed)
    
    max_seq_len = resolve_max_seq_len(config.max_history_minutes)
    max_seq_len_actual = max_seq_len or MAX_SEQ_LEN
    
    train_ds = TransformerOnlyDataset(train_df, feature_cols, label_cols, eligibility_cols, max_seq_len)
    val_ds = TransformerOnlyDataset(val_df, feature_cols, label_cols, eligibility_cols, max_seq_len)
    test_ds = TransformerOnlyDataset(test_df, feature_cols, label_cols, eligibility_cols, max_seq_len)
    
    if len(train_ds) == 0 or len(val_ds) == 0:
        print("Not enough data for training/validation. Skipping.")
        return None
    
    max_seq_len_actual = max(train_ds.max_seq_len, val_ds.max_seq_len, test_ds.max_seq_len, MAX_SEQ_LEN)
    
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, 
                             collate_fn=collate_transformer_only, num_workers=NUM_WORKERS, 
                             pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, 
                           collate_fn=collate_transformer_only, num_workers=NUM_WORKERS, 
                           pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False, 
                            collate_fn=collate_transformer_only, num_workers=NUM_WORKERS, 
                            pin_memory=torch.cuda.is_available())
    
    wandb_run = init_wandb_run(
        name="transformer-only-run",
        job_type="train",
        tags=["transformer-only"],
        config={**asdict(config), "feature_dim": len(feature_cols), "num_labels": len(label_cols)},
    )
    
    model = TransformerOnlyModel(
        sequence_feature_dim=len(feature_cols),
        d_model=config.d_model,
        n_layers=config.n_layers,
        d_ff_multiplier=config.d_ff_multiplier,
        transformer_dropout=config.transformer_dropout,
        max_seq_len=max_seq_len_actual,
        num_labels=len(label_cols),
    ).to(DEVICE)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    total_steps = max(1, config.max_epochs * len(train_loader))
    scheduler = create_scheduler(optimizer, total_steps)
    
    best_state = None
    best_val_metric = -float("inf")
    best_epoch = -1
    best_val_metrics = None
    patience_counter = 0
    
    for epoch in range(1, config.max_epochs + 1):
        model.train()
        epoch_train_losses = []
        
        for batch in train_loader:
            seq_inputs = batch["sequence_inputs"].to(DEVICE)
            labels_full = batch["labels"].to(DEVICE)
            eligibility_full = batch["eligibility"].to(DEVICE)
            mask = batch["mask"].to(DEVICE)
            valid_lengths = batch["valid_lengths"].to(DEVICE)
            
            optimizer.zero_grad(set_to_none=True)
            
            logits = model(seq_inputs, src_key_padding_mask=(mask == 0), valid_lengths=valid_lengths)
            
            B = seq_inputs.shape[0]
            last_indices = (valid_lengths - 1).clamp(min=0)
            batch_indices = torch.arange(B, device=DEVICE)
            labels = labels_full[batch_indices, last_indices]
            eligibility = eligibility_full[batch_indices, last_indices]
            valid_mask = eligibility
            
            loss = compute_loss(logits, labels, valid_mask, pos_weight, config.label_smoothing)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            epoch_train_losses.append(loss.item())
        
        val_metrics = evaluate_model(model, val_loader, pos_weight, len(label_cols), DEVICE, 
                                     label_smoothing=config.label_smoothing)
        metric = val_metrics["macro_pr_auc"]
        
        improved = metric > best_val_metric + 1e-4
        if improved:
            best_val_metric = metric
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            best_epoch = epoch
            best_val_metrics = val_metrics.copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                break
        
        if wandb_run:
            wandb_run.log({
                "epoch": epoch,
                "train/loss": float(np.mean(epoch_train_losses)) if epoch_train_losses else float("nan"),
                "val/loss": val_metrics["loss"],
                "val/macro_pr_auc": val_metrics["macro_pr_auc"],
                "train/learning_rate": scheduler.get_last_lr()[0],
            }, step=epoch)
    
    if best_state is None:
        best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        best_epoch = epoch
    
    model.load_state_dict(best_state)
    test_metrics = evaluate_model(model, test_loader, pos_weight, len(label_cols), DEVICE, 
                                  label_smoothing=config.label_smoothing)
    
    #save checkpoint
    model_path = OUTPUT_DIR / "best_model.pt"
    torch.save({
        "config": asdict(config),
        "state_dict": best_state,
        "val_macro_pr_auc": best_val_metric,
        "test_metrics": test_metrics,
    }, model_path)
    
    summary_path = OUTPUT_DIR / "summary.json"
    with open(summary_path, "w") as fp:
        json.dump({
            "config": asdict(config),
            "best_epoch": best_epoch,
            "val_macro_pr_auc": best_val_metric,
            "best_val_metrics": best_val_metrics,
            "test_metrics": test_metrics,
        }, fp, indent=2)
    
    if wandb_run:
        wandb_run.log({
            "final/val_macro_pr_auc": best_val_metric,
            "final/test_macro_pr_auc": test_metrics["macro_pr_auc"],
        })
        wandb_run.finish()
    
    print(f"Best epoch: {best_epoch}, Val PR-AUC: {best_val_metric:.4f}, Test PR-AUC: {test_metrics['macro_pr_auc']:.4f}")
    
    return {
        "best_epoch": best_epoch,
        "val_macro_pr_auc": best_val_metric,
        "test_macro_pr_auc": test_metrics["macro_pr_auc"],
        "test_metrics": test_metrics,
    }


def main():
    parser = argparse.ArgumentParser(description="Transformer-Only Model Training")
    parser.add_argument("--epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--d-ff-multiplier", type=float, default=3.0)
    parser.add_argument("--transformer-dropout", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--label-smoothing", type=float, default=0.0) 
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--data-limit", type=int)
    args = parser.parse_args()
    
    print("Loading data")
    
    is_single_file = TEAM_SEQUENCE_FEATURES_PATH.is_file() and TEAM_SEQUENCE_FEATURES_PATH.suffix == ".parquet"
    is_directory = TEAM_SEQUENCE_FEATURES_PATH.is_dir()
    
    if not is_single_file and not is_directory:
        print(f"Error: Sequence data not found at {TEAM_SEQUENCE_FEATURES_PATH}")
        print("Expected either a single .parquet file (e.g., godview_cleaned.parquet) or a directory with partitioned parquet files.")
        print("Please run 'python preprocess_all.py' first or set TEAM_SEQUENCE_FEATURES_PATH environment variable.")
        return
    
    if args.data_limit and not is_single_file:
        print(f"Optimized loading: Collecting {args.data_limit} unique games from partitions...")
        partition_files = sorted(TEAM_SEQUENCE_FEATURES_PATH.glob("part_*.parquet"))
        if not partition_files:
            raise FileNotFoundError(f"No partition files found in {TEAM_SEQUENCE_FEATURES_PATH}")
        
        #scan partitions to collect unique matchIds until we have enough
        target_match_ids = []
        partitions_to_load = []
        seen_match_ids = set()
        
        for part_file in partition_files:
            #read only matchId column 
            part_match_ids = pd.read_parquet(part_file, columns=["matchId"])["matchId"].unique()
            
            #track which partition we need to load
            partition_needed = False
            
            #add new matchIds in order 
            for mid in part_match_ids:
                if mid not in seen_match_ids:
                    target_match_ids.append(mid)
                    seen_match_ids.add(mid)
                    partition_needed = True
                    
                    #stop once we have enough unique matchIds
                    if len(target_match_ids) >= args.data_limit:
                        #take exactly args.data_limit matchIds
                        target_match_ids = target_match_ids[:args.data_limit]
                        seen_match_ids = set(target_match_ids)
                        break
            
            if partition_needed:
                partitions_to_load.append(part_file)
            
            #stop scanning if we have enough
            if len(target_match_ids) >= args.data_limit:
                break
        
        target_match_ids = set(target_match_ids)
        
        if len(target_match_ids) < args.data_limit:
            print(f"Warning: Only found {len(target_match_ids)} unique games, requested {args.data_limit}")
        
        print(f"Found {len(target_match_ids)} unique games in {len(partitions_to_load)} partition(s)")
        print(f"Loading partitions: {[f.name for f in partitions_to_load]}")
        
        #load only the needed partitions
        dfs = []
        for part_file in partitions_to_load:
            df_part = pd.read_parquet(part_file)
            #filter to target matchIds immediately 
            df_part = df_part[df_part["matchId"].isin(target_match_ids)]
            if len(df_part) > 0:
                dfs.append(df_part)
        
        if not dfs:
            raise ValueError(f"No data found for the requested {args.data_limit} games")
        
        #concatenate and handle schema drift
        team_sequence_df = pd.concat(dfs, ignore_index=True)
        del dfs  
        
        print(f"Filling NaN values from schema drift (shape before: {team_sequence_df.shape})")
        team_sequence_df = team_sequence_df.fillna(0)
        print(f"Shape after fillna: {team_sequence_df.shape}")
        
        #ensure we only have the target matchIds 
        team_sequence_df = team_sequence_df[team_sequence_df["matchId"].isin(target_match_ids)]
        print(f"Loaded {len(team_sequence_df):,} rows for {len(team_sequence_df['matchId'].unique()):,} games")
    elif is_single_file:
        #load directly from parquet file
        print(f"Loading single parquet file from {TEAM_SEQUENCE_FEATURES_PATH}...")
        team_sequence_df = pd.read_parquet(TEAM_SEQUENCE_FEATURES_PATH)
        
        #fill NaNs with 0 
        print(f"Filling NaN values from schema drift (shape before: {team_sequence_df.shape})...")
        team_sequence_df = team_sequence_df.fillna(0)
        print(f"Shape after fillna: {team_sequence_df.shape}")
        
        if args.data_limit:
            match_ids = team_sequence_df["matchId"].unique()[:args.data_limit]
            team_sequence_df = team_sequence_df[team_sequence_df["matchId"].isin(match_ids)]
            print(f"Limited to {args.data_limit} games: {len(team_sequence_df):,} rows, {len(match_ids):,} unique games")
    else:
        #load all partitions 
        print("Loading all partitioned parquet files...")
        team_sequence_df = pd.read_parquet(TEAM_SEQUENCE_FEATURES_PATH)
        print(f"Filling NaN values from schema drift (shape before: {team_sequence_df.shape})")
        team_sequence_df = team_sequence_df.fillna(0)
        print(f"Shape after fillna: {team_sequence_df.shape}")
    
    with open(TEAM_SEQUENCE_METADATA_PATH, "r") as meta_file:
        team_sequence_meta = json.load(meta_file)
    
    team_label_cols = team_sequence_meta["team_label_cols"]
    team_eligibility_cols = team_sequence_meta.get("team_eligibility_cols", [])
    team_sequence_feature_cols = team_sequence_meta["team_sequence_feature_cols"]
    
    EXPECTED_LABEL_ORDER = [
        "dragon_taken_next_1min",
        "dragon_taken_next_2min",
        "dragon_taken_next_3min",
        "baron_taken_next_1min",
        "baron_taken_next_2min",
        "baron_taken_next_3min",
    ]
    if team_label_cols != EXPECTED_LABEL_ORDER:
        raise ValueError(
            f"Label order mismatch! This will cause logits[i] to correspond to wrong label.\n"
            f"Expected order (from TransformerOnlyModel.forward): {EXPECTED_LABEL_ORDER}\n"
            f"Actual order (from metadata): {team_label_cols}\n"
            f"Either fix the metadata preprocessing to match this order, or update TransformerOnlyModel.forward's "
            f"concatenation logic to match the actual label order."
        )
    
    #train/val/test split
    match_ids = team_sequence_df["matchId"].unique()
    train_ids, temp_ids = train_test_split(match_ids, test_size=VAL_SPLIT + TEST_SPLIT, random_state=RANDOM_SEED)
    val_ids, test_ids = train_test_split(temp_ids, test_size=TEST_SPLIT / (VAL_SPLIT + TEST_SPLIT), random_state=RANDOM_SEED)
    
    train_df = team_sequence_df[team_sequence_df["matchId"].isin(train_ids)]
    val_df = team_sequence_df[team_sequence_df["matchId"].isin(val_ids)]
    test_df = team_sequence_df[team_sequence_df["matchId"].isin(test_ids)]
    
    #compute pos_weight
    pos_weights = []
    for label_col in team_label_cols:
        elig_col = None
        if team_eligibility_cols:
            idx = team_label_cols.index(label_col)
            if idx < len(team_eligibility_cols):
                elig_col = team_eligibility_cols[idx]
        pos_weights.append(_compute_pos_weight(train_df, label_col, elig_col))
    pos_weight = torch.tensor(pos_weights, dtype=torch.float32).to(DEVICE)
    
    config = TrainConfig(
        d_model=args.d_model,
        n_layers=args.n_layers,
        d_ff_multiplier=args.d_ff_multiplier,
        transformer_dropout=args.transformer_dropout,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        max_epochs=args.epochs,
        label_smoothing=args.label_smoothing,
        seed=args.seed,
    )
    
    result = train_single_config(
        config, train_df, val_df, test_df,
        team_sequence_feature_cols, team_label_cols, team_eligibility_cols, pos_weight
    )
    
    if result:
        print(f"\nTraining complete!")
        print(f"Val PR-AUC: {result['val_macro_pr_auc']:.4f}")
        print(f"Test PR-AUC: {result['test_macro_pr_auc']:.4f}")


if __name__ == "__main__":
    main()

