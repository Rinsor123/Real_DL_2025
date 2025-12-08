"""
Hybrid Transformer-CNN Model for Objective Prediction

we decided to go with scripts rather than notebooks for the main part as this was way easier to work with on LUMI
"""

import argparse
import fcntl
import json
import math
import os
import random
import re
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Tuple, Dict, List, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import gc
import optuna
import wandb 
from optuna.samplers import TPESampler


#in order to match spatial and tabular match data
def _hash_match_id(match_id: str) -> int:
    """
    convert matchId string to int32 hash 
    """
    return hash(match_id) & 0x7FFFFFFF


def _env_flag(name: str, default: str = "0") -> bool:
    val = os.getenv(name, default)
    return val is not None and val.lower() not in ("0", "false", "no", "")


def _log(msg: str):
    print(msg)


USE_WANDB = _env_flag("USE_WANDB", "1")
WANDB_MODE = os.getenv("WANDB_MODE", "offline")
WANDB_PROJECT = os.getenv("WANDB_PROJECT", "hybrid-objective-model")
WANDB_ENTITY = os.getenv("WANDB_ENTITY")

#config
DATA_ROOT = Path(os.getenv("DATA_ROOT", "/project/project_465002423/Deep-Learning-Project-2025/combined_data"))
SPATIAL_PACKED_PATH = DATA_ROOT / "spatial_data_packed.pt"
SPATIAL_DATA_PACKED_PATH = SPATIAL_PACKED_PATH
TEAM_SEQUENCE_METADATA_PATH = DATA_ROOT / "team_sequence_metadata.json"
SEQ_PATH = Path(os.getenv("TEAM_SEQUENCE_FEATURES_PATH", str(DATA_ROOT / "godview_cleaned.parquet")))


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

MAP_SIZE = 15000
GRID_SIZE = 64

TOWER_LOCATIONS = [
    (8955, 8510), (9767, 10113), (11134, 11207), (11593, 11669),
    (13052, 12612), (12611, 13084), (5846, 6396), (5048, 4812),
    (3651, 3696), (3210, 3217), (2177, 1807), (1748, 2270),
    (4318, 13875), (981, 10441), (7943, 13411), (1512, 6699),
    (1169, 4287), (1172, 3583), (10481, 13650), (11275, 13657),
    (11275, 13663), (13866, 4505), (10504, 1029), (13327, 8226),
    (13624, 10572), (6919, 1483), (13599, 11319), (4281, 1253),
    (3468, 1230), (13594, 11319)
]

SEARCH_OUTPUT_DIR = DATA_ROOT / "hybrid_model_search"
SEARCH_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

HYPERPARAM_SEARCH_TRIALS = 24
HYPERPARAM_SEARCH_SEED = 1337

@dataclass
class TrainConfig:
    d_model: int = 128
    n_layers: int = 2
    d_ff_multiplier: float = 3.0
    transformer_dropout: float = 0.1
    max_history_minutes: str | int = "all"

    cnn_channels: int = 64
    cnn_dropout: float = 0.1
    spatial_sigma: float = 1.5  # spatial decay rate for heatmap blurring

    fusion_hidden_dim: int = 256
    fusion_dropout: float = 0.2

    batch_size: int = BATCH_SIZE
    lr: float = 1e-3
    weight_decay: float = WEIGHT_DECAY
    max_epochs: int = MAX_EPOCHS
    patience: int = EARLY_STOPPING_PATIENCE
    label_smoothing: float = 0.0

    seed: int = RANDOM_SEED

class SpatialInputLayer(nn.Module):
    """
    renders raw game state into multi-channel heatmap on GPU
    works!!
    """
    def __init__(self, grid_size=GRID_SIZE, map_size=MAP_SIZE, sigma=1.5, turret_range=1100):
        super().__init__()
        self.grid_size = grid_size
        self.scale = map_size / grid_size
        self.sigma = sigma
        self.turret_range_grid = int(turret_range / self.scale)

        self.register_buffer('tower_locs', torch.tensor(TOWER_LOCATIONS, dtype=torch.float32))

        r = self.turret_range_grid
        y_coords, x_coords = torch.meshgrid(
            torch.arange(-r, r + 1, dtype=torch.float32),
            torch.arange(-r, r + 1, dtype=torch.float32),
            indexing='ij'
        )
        range_mask = (x_coords**2 + y_coords**2 <= r**2).float()
        self.register_buffer('range_mask', range_mask)

        kernel_size = int(2 * 4 * sigma + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
        x = torch.arange(kernel_size) - kernel_size // 2
        x = x.float()
        k1d = torch.exp(-x**2 / (2 * sigma**2))
        kernel = k1d[:, None] * k1d[None, :]
        kernel = kernel / kernel.sum()

        self.num_channels = 16
        self.register_buffer('kernel', kernel.expand(self.num_channels, 1, kernel_size, kernel_size))
        self.padding = kernel_size // 2

    def _to_grid(self, coords):
        return (coords / self.scale).long().clamp(0, self.grid_size - 1)

    def forward(self, player_trails, dead_towers_mask, kill_ctx, obj_status):
        B = dead_towers_mask.shape[0] #get back to this
        H, W = self.grid_size, self.grid_size

        canvas = torch.zeros(B, self.num_channels, H, W, device=dead_towers_mask.device)

        tower_grid = self._to_grid(self.tower_locs)
        tx, ty = tower_grid[:, 0], tower_grid[:, 1]
        r = self.turret_range_grid
        mask_size = 2 * r + 1

        for t_idx in range(30):
            is_dead = dead_towers_mask[:, t_idx] > 0.5
            is_alive = ~is_dead
            cx, cy = int(tx[t_idx]), int(ty[t_idx])

            canvas[is_alive, 0, cy, cx] = 1.0
            canvas[is_dead, 1, cy, cx] = 1.0

            if is_alive.any():
                x0, x1 = max(0, cx - r), min(W, cx + r + 1)
                y0, y1 = max(0, cy - r), min(H, cy + r + 1)
                mx0 = max(0, r - cx)
                mx1 = mask_size - max(0, (cx + r + 1) - W)
                my0 = max(0, r - cy)
                my1 = mask_size - max(0, (cy + r + 1) - H)
                mask_slice = self.range_mask[my0:my1, mx0:mx1]
                canvas[is_alive, 2, y0:y1, x0:x1] = torch.maximum(
                    canvas[is_alive, 2, y0:y1, x0:x1],
                    mask_slice.unsqueeze(0).expand(is_alive.sum(), -1, -1)
                )

        drag_pos = self._to_grid(torch.tensor([10000., 5000.], device=dead_towers_mask.device))
        baron_pos = self._to_grid(torch.tensor([5000., 10000.], device=dead_towers_mask.device))
        has_drag = obj_status[:, 0] > 0.5
        has_baron = obj_status[:, 1] > 0.5
        canvas[has_drag, 3, drag_pos[1], drag_pos[0]] = 1.0
        canvas[has_baron, 3, baron_pos[1], baron_pos[0]] = 1.0

        k_coords, k_vals, k_teams, k_b_idx = kill_ctx
        if k_coords.shape[0] > 0:
            k_grid = self._to_grid(k_coords)
            channels = 4 + k_teams.long()
            indices = (k_b_idx, channels, k_grid[:, 1], k_grid[:, 0])
            canvas.index_put_(indices, k_vals, accumulate=True)

        p_coords, p_vals, p_b_idx, p_channels = player_trails
        if p_coords.shape[0] > 0:
            p_grid = self._to_grid(p_coords)
            channels = 6 + p_channels
            indices = (p_b_idx, channels, p_grid[:, 1], p_grid[:, 0])
            canvas.index_put_(indices, p_vals, accumulate=True)

        blurred = F.conv2d(canvas, self.kernel, padding=self.padding, groups=self.num_channels)
        return blurred

class BasicResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class SpatialCNN(nn.Module):
    """"
    ResNet-style CNN for spatial heatmap processing
    allows tuning now
    """
    def __init__(self, base_channels=64, dropout=0.1, spatial_sigma=1.5):
        super().__init__()
        self.renderer = SpatialInputLayer(grid_size=GRID_SIZE, map_size=MAP_SIZE, sigma=spatial_sigma)

        self.inplanes = base_channels
        self.conv1 = nn.Conv2d(16, base_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(base_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(base_channels, 2)
        self.layer2 = self._make_layer(base_channels * 2, 2, stride=2)
        self.layer3 = self._make_layer(base_channels * 4, 2, stride=2)
        self.layer4 = self._make_layer(base_channels * 8, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout)

        self.embed_dim = base_channels * 8

    def _make_layer(self, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, 1, stride, bias=False),
                nn.BatchNorm2d(planes),
            )
        layers = [BasicResBlock(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(BasicResBlock(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, player_trails, dead_towers, kill_ctx, obj_status):
        x = self.renderer(player_trails, dead_towers, kill_ctx, obj_status)

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
        x = self.dropout(x)

        return x  #(B, embed_dim)

class TransformerEncoder(nn.Module):
    """
    transformer encoder for temporal team features
    tuning also works!
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

class HybridModel(nn.Module):
    def __init__(
        self,
        sequence_feature_dim: int,
        d_model: int = 128,
        n_layers: int = 2,
        d_ff_multiplier: float = 3.0,
        transformer_dropout: float = 0.1,
        cnn_channels: int = 64,
        cnn_dropout: float = 0.1,
        spatial_sigma: float = 1.5,
        fusion_hidden_dim: int = 256,
        fusion_dropout: float = 0.2,
        max_seq_len: int = 64,
        num_labels: int = 2,
    ):
        super().__init__()

        self.transformer = TransformerEncoder(
            feature_dim=sequence_feature_dim,
            d_model=d_model,
            nhead=4,
            num_layers=n_layers,
            max_seq_len=max_seq_len,
            dim_feedforward=int(d_model * d_ff_multiplier),
            dropout=transformer_dropout,
        )

        self.cnn = SpatialCNN(
            base_channels=cnn_channels,
            dropout=cnn_dropout,
            spatial_sigma=spatial_sigma,
        )

        cnn_embed_dim = cnn_channels * 8
        fusion_input_dim = d_model + cnn_embed_dim

        self.shared_fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(fusion_dropout),
            nn.Linear(fusion_hidden_dim, fusion_hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(fusion_dropout),
        )

        assert num_labels == 6, "6 labels: 3 horizons (1min, 2min, 3min) × 2 objectives (dragon, baron)"
        self.head_1min = nn.Linear(fusion_hidden_dim // 2, 2)  # dragon_1min, baron_1min
        self.head_2min = nn.Linear(fusion_hidden_dim // 2, 2)  # dragon_2min, baron_2min
        self.head_3min = nn.Linear(fusion_hidden_dim // 2, 2)  # dragon_3min, baron_3min

        self.max_seq_len = max_seq_len
        self.num_labels = num_labels

    def forward(
        self,
        sequence_features: torch.Tensor,
        spatial_data: Dict,
        src_key_padding_mask: torch.Tensor | None = None,
        valid_lengths: torch.Tensor = None,
    ):
        """
        many-to-One forward pass
        """
        B, T, _ = sequence_features.shape
        device = sequence_features.device

        transformer_out = self.transformer(sequence_features, src_key_padding_mask)  #(B, T, d_model) Er det rigtigt?

        if valid_lengths is None:
            valid_lengths = torch.full((B,), T, dtype=torch.long, device=device)

        last_indices = (valid_lengths - 1).clamp(min=0)  #(B,)  #many-to-one, use last valid timestep as summary
        batch_indices = torch.arange(B, device=device)
        transformer_last = transformer_out[batch_indices, last_indices]  #(B, d_model)

        cnn_out = self.cnn(
            spatial_data["player_trails"],
            spatial_data["dead_towers"],
            spatial_data["kill_ctx"],
            spatial_data["obj_status"],
        )  #(B, cnn_embed_dim)

        fused = torch.cat([transformer_last, cnn_out], dim=-1)  #(B, fusion_input_dim)

        shared_features = self.shared_fusion(fused)  #(B, fusion_hidden_dim // 2)

        logits_1min = self.head_1min(shared_features)  #(B, 2): [dragon_1min, baron_1min]
        logits_2min = self.head_2min(shared_features)  #(B, 2): [dragon_2min, baron_2min]
        logits_3min = self.head_3min(shared_features)  #(B, 2): [dragon_3min, baron_3min]

        logits = torch.cat([
            logits_1min[:, 0:1],  #dragon_1min
            logits_2min[:, 0:1],  #dragon_2min
            logits_3min[:, 0:1],  #dragon_3min
            logits_1min[:, 1:2],  #baron_1min
            logits_2min[:, 1:2],  #baron_2min
            logits_3min[:, 1:2],  #baron_3min
        ], dim=-1)  #(B, 6)  #many-to-one label order: dragon1/2/3, baron1/2/3

        return logits


# dataset

@dataclass
class HybridBatch:
    sequence_inputs: torch.Tensor  #(T, F)
    labels: torch.Tensor  #(T, num_labels)
    eligibility: torch.Tensor  #(T, num_labels)
    valid_len: int

    spatial_p_coords: torch.Tensor
    spatial_p_vals: torch.Tensor
    spatial_p_channels: torch.Tensor
    spatial_dead_towers: torch.Tensor
    spatial_obj_status: torch.Tensor
    spatial_kills: torch.Tensor


class HybridDataset(Dataset):
    """
    dataset that provides both sequence features AND spatial data for the final frame
    jacobs rod er væk...
    """
    def __init__(
        self,
        sequence_df: pd.DataFrame,
        spatial_data: List[Dict] | Dict[str, torch.Tensor],
        feature_cols: List[str],
        label_cols: List[str],
        eligibility_cols: List[str] | None = None,
        max_seq_len: int | None = MAX_SEQ_LEN,
        stride: int = 1,
        min_history: int = 1,
    ):
        self.packed_data = spatial_data
        self.meta = spatial_data["meta"]  #[N, 2] 
        self.obj_status = spatial_data["obj_status"]  #[N, 2]
        self.towers = spatial_data["towers"]  # [N, 30] 
        self.kills_val = spatial_data["kills_val"]  # [M, 4]
        self.kills_idx = spatial_data["kills_idx"]  # [N, 2] 
        self.trails_val = spatial_data["trails_val"]  # [P, 3]
        self.trails_idx = spatial_data["trails_idx"]  # [N, 10, 2]

        meta_np = self.meta.numpy()
        self.spatial_lookup = {}
        for i in range(len(meta_np)):
            match_hash = int(meta_np[i, 0])
            minute = int(meta_np[i, 1])
            self.spatial_lookup[(match_hash, minute)] = i

        grouped = sequence_df.sort_values(["matchId", "teamId", "minute"])
        sequences = []
        eligibility_cols = eligibility_cols or []
        requested_max_seq_len = max_seq_len

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

            max_len = requested_max_seq_len or len(arr)

            for end_idx in range(min_history, len(arr) + 1, stride):
                start_idx = max(0, end_idx - max_len)

                seq = arr[start_idx:end_idx]
                lbl = labels[start_idx:end_idx]
                elig_slice = elig[start_idx:end_idx]
                chunk_minutes = minutes[start_idx:end_idx]

                final_minute = int(chunk_minutes[-1])

                match_hash = _hash_match_id(match_id)
                spatial_key = (match_hash, final_minute)
                if spatial_key in self.spatial_lookup:
                    sequences.append({
                        "seq": seq,
                        "labels": lbl,
                        "elig": elig_slice,
                        "match_id": match_id,
                        "match_hash": match_hash,
                        "team_id": team_id,
                        "final_minute": final_minute,
                    })

        self.sequences = sequences
        if requested_max_seq_len is None:
            self.max_seq_len = max((len(s["seq"]) for s in sequences), default=0)
        else:
            self.max_seq_len = requested_max_seq_len
        self.feature_cols = feature_cols
        self.feature_dim = len(feature_cols)
        self.stride = stride

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict:
        item = self.sequences[idx]
        seq, lbl, elig = item["seq"], item["labels"], item["elig"]
        valid_len = len(seq)

        pad_len = max(self.max_seq_len - valid_len, 0) #we need to pad the sequence to the max length
        if pad_len > 0:
            seq = np.pad(seq, ((0, pad_len), (0, 0)), mode="constant")
            lbl = np.pad(lbl, ((0, pad_len), (0, 0)), mode="constant")
            elig = np.pad(elig, ((0, pad_len), (0, 0)), mode="constant")
        elif pad_len < 0:
            seq = seq[:self.max_seq_len]
            lbl = lbl[:self.max_seq_len]
            elig = elig[:self.max_seq_len]
            valid_len = self.max_seq_len

        team_id = item["team_id"]

        match_hash = item["match_hash"]
        spatial_key = (match_hash, item["final_minute"])
        spatial_idx = self.spatial_lookup[spatial_key]
        spatial_dict = self._process_spatial_state_packed(spatial_idx, team_id)

        return {
            "sequence_inputs": torch.from_numpy(seq),
            "labels": torch.from_numpy(lbl),
            "eligibility": torch.from_numpy(elig),
            "valid_len": valid_len,
            **spatial_dict,
        }

    def _process_spatial_state_packed(self, idx: int, team_id: int) -> Dict:
        """
        process spatial state from packed tensors by index
        """
        dead_towers = self.towers[idx].to(torch.float32)

        obj_status = self.obj_status[idx].to(torch.float32)

        kill_start = int(self.kills_idx[idx, 0])
        kill_count = int(self.kills_idx[idx, 1])
        if kill_count > 0:
            kills = self.kills_val[kill_start:kill_start + kill_count].to(torch.float32)
        else:
            kills = torch.empty((0, 4), dtype=torch.float32)

        p_coords_list = []
        p_vals_list = []
        p_channels_list = []

        for player_idx in range(10):  #10 players 
            trail_start = int(self.trails_idx[idx, player_idx, 0])
            trail_count = int(self.trails_idx[idx, player_idx, 1])
            if trail_count > 0:
                trail_slice = self.trails_val[trail_start:trail_start + trail_count].to(torch.float32)
                p_coords_list.extend(trail_slice[:, :2].tolist())  
                p_vals_list.extend(trail_slice[:, 2].tolist())  
                p_channels_list.extend([player_idx] * trail_count)

        if p_coords_list:
            p_coords = torch.tensor(p_coords_list, dtype=torch.float32)
            p_vals = torch.tensor(p_vals_list, dtype=torch.float32)
            p_channels = torch.tensor(p_channels_list, dtype=torch.long)
        else:
            p_coords = torch.empty((0, 2), dtype=torch.float32)
            p_vals = torch.empty((0,), dtype=torch.float32)
            p_channels = torch.empty((0,), dtype=torch.long)

        return {
            "spatial_p_coords": p_coords,
            "spatial_p_vals": p_vals,
            "spatial_p_channels": p_channels,
            "spatial_dead_towers": dead_towers,
            "spatial_obj_status": obj_status,
            "spatial_kills": kills,
            "team_id": team_id,
        }


def collate_hybrid(batch):
    """
    custom collate function for hybrid dataset
    is this correct?
    """
    sequence_inputs = torch.stack([item["sequence_inputs"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    eligibility = torch.stack([item["eligibility"] for item in batch])
    lengths = torch.tensor([item["valid_len"] for item in batch], dtype=torch.long)
    mask = (torch.arange(sequence_inputs.shape[1]).unsqueeze(0) < lengths.unsqueeze(1)).float()   #mask padding for transformer

    dead_towers = torch.stack([item["spatial_dead_towers"] for item in batch])
    obj_status = torch.stack([item["spatial_obj_status"] for item in batch])

    all_kill_coords = []
    all_kill_vals = []
    all_kill_teams = []
    all_kill_b_idx = []

    for i, item in enumerate(batch):
        k = item["spatial_kills"]
        if k.shape[0] > 0:
            all_kill_coords.append(k[:, 0:2])
            all_kill_teams.append(k[:, 2])
            all_kill_vals.append(k[:, 3])
            all_kill_b_idx.append(torch.full((k.shape[0],), i, dtype=torch.long))

    if all_kill_coords:
        k_coords = torch.cat(all_kill_coords)
        k_vals = torch.cat(all_kill_vals)
        k_teams = torch.cat(all_kill_teams)
        k_b_idx = torch.cat(all_kill_b_idx)
    else:
        k_coords = torch.empty((0, 2))
        k_vals = torch.empty((0,))
        k_teams = torch.empty((0,))
        k_b_idx = torch.empty((0,), dtype=torch.long)

    all_p_coords = []
    all_p_vals = []
    all_p_channels = []
    all_p_b_idx = []

    for i, item in enumerate(batch):
        pc = item["spatial_p_coords"]
        pv = item["spatial_p_vals"]
        pch = item["spatial_p_channels"]

        if pc.shape[0] > 0:
            all_p_coords.append(pc)
            all_p_vals.append(pv)
            all_p_channels.append(pch)
            all_p_b_idx.append(torch.full((pc.shape[0],), i, dtype=torch.long))

    if all_p_coords:
        p_coords = torch.cat(all_p_coords)
        p_vals = torch.cat(all_p_vals)
        p_channels = torch.cat(all_p_channels)
        p_b_idx = torch.cat(all_p_b_idx)
    else:
        p_coords = torch.empty((0, 2))
        p_vals = torch.empty((0,))
        p_channels = torch.empty((0,), dtype=torch.long)
        p_b_idx = torch.empty((0,), dtype=torch.long)

    return {
        "sequence_inputs": sequence_inputs,
        "labels": labels,
        "eligibility": eligibility,
        "mask": mask,
        "valid_lengths": lengths,
        "spatial_data": {
            "player_trails": (p_coords, p_vals, p_b_idx, p_channels),
            "dead_towers": dead_towers,
            "obj_status": obj_status,
            "kill_ctx": (k_coords, k_vals, k_teams, k_b_idx),
        },
    }

def _detect_continuous_features(dataframe: pd.DataFrame, feature_cols: List[str]) -> List[str]:
    continuous = []
    for col in feature_cols:
        series = dataframe[col]
        if not np.issubdtype(series.dtype, np.number):
            continue
        unique_values = series.dropna().unique()
        if len(unique_values) <= 2 and set(np.round(unique_values, 6)).issubset({0.0, 1.0}):
            continue
        continuous.append(col)
    return continuous

def _compute_normalization_stats(df: pd.DataFrame, columns: List[str]) -> Dict[str, Tuple[float, float]]:
    stats = {}
    for col in columns:
        mean = float(df[col].mean())
        std = float(df[col].std())
        if not np.isfinite(std) or std < 1e-6:
            std = 1.0
        stats[col] = (mean, std)
    return stats

def _apply_normalization(df: pd.DataFrame, stats: Dict[str, Tuple[float, float]]) -> pd.DataFrame:
    df_norm = df.copy()
    updates = {}
    for col, (mean, std) in stats.items():
        if col in df_norm.columns:
            normalized_values = ((df_norm[col].astype(np.float32) - mean) / std).astype(np.float32)
            normalized_values = normalized_values.fillna(0.0)
            updates[col] = normalized_values
    if updates:
        df_norm = df_norm.assign(**updates)
    numeric_cols = df_norm.select_dtypes(include=[np.number]).columns
    df_norm[numeric_cols] = df_norm[numeric_cols].fillna(0.0)
    return df_norm

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

#training
def format_label_for_logging(label_name: str) -> str:
    """
    convert label name 
    """
    match = re.match(r'^(\w+)_taken_next_(\d+)min$', label_name)
    if match:
        base = match.group(1)
        horizon = match.group(2)  
        return f"{base}_{horizon}min"
    return label_name


def compute_horizon_avg_pr_auc(per_label_pr_auc: List[float]) -> Dict[str, float]:
    """
    compute average PR-AUC for each horizon (1min, 2min, 3min)
    seems ok?
    """
    if len(per_label_pr_auc) != 6:
        return {
            "avg_pr_auc_1min": float("nan"),
            "avg_pr_auc_2min": float("nan"),
            "avg_pr_auc_3min": float("nan"),
        }

    pr_1min = [per_label_pr_auc[0], per_label_pr_auc[3]]  # dragon_1min, baron_1min
    pr_2min = [per_label_pr_auc[1], per_label_pr_auc[4]]  # dragon_2min, baron_2min
    pr_3min = [per_label_pr_auc[2], per_label_pr_auc[5]]  # dragon_3min, baron_3min

    def nanmean(values):
        valid = [v for v in values if not np.isnan(v)]
        return float(np.mean(valid)) if valid else float("nan")

    return {
        "avg_pr_auc_1min": nanmean(pr_1min),
        "avg_pr_auc_2min": nanmean(pr_2min),
        "avg_pr_auc_3min": nanmean(pr_3min),
    }


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
    """
    evaluate model with many-to-one architecture
    doesnt seem to data leak anymore
    """
    model.eval()
    total_loss = 0.0
    total_weight = 0.0
    y_true = [[] for _ in range(num_labels)]
    y_score = [[] for _ in range(num_labels)]

    with torch.no_grad():
        for batch in loader:
            seq_inputs = batch["sequence_inputs"].to(device)
            labels_full = batch["labels"].to(device)  #(B, T, num_labels)
            eligibility_full = batch["eligibility"].to(device)  #(B, T, num_labels)
            mask = batch["mask"].to(device)
            valid_lengths = batch["valid_lengths"].to(device)  #(B,)

            spatial_data = {
                "player_trails": tuple(t.to(device) for t in batch["spatial_data"]["player_trails"]),
                "dead_towers": batch["spatial_data"]["dead_towers"].to(device),
                "obj_status": batch["spatial_data"]["obj_status"].to(device),
                "kill_ctx": tuple(t.to(device) for t in batch["spatial_data"]["kill_ctx"]),
            }

            logits = model(seq_inputs, spatial_data, src_key_padding_mask=(mask == 0), valid_lengths=valid_lengths)

            logits = torch.clamp(logits, min=-100, max=100)
            logits = torch.where(torch.isnan(logits), torch.zeros_like(logits), logits)

            B = seq_inputs.shape[0]
            last_indices = (valid_lengths - 1).clamp(min=0)  #(B,)
            batch_indices = torch.arange(B, device=device)
            labels = labels_full[batch_indices, last_indices]  #(B, num_labels)
            eligibility = eligibility_full[batch_indices, last_indices]  #(B, num_labels)

            preds = torch.sigmoid(logits) * eligibility  #(B, num_labels)
            valid_mask = eligibility  #(B, num_labels)

            batch_loss = compute_loss(logits, labels, valid_mask, pos_weight, label_smoothing)
            batch_weight = valid_mask.sum().item()
            total_loss += batch_loss.item() * batch_weight
            total_weight += batch_weight

            valid_bool = valid_mask.bool()  #(B, num_labels)
            for idx in range(num_labels):
                idx_mask = valid_bool[:, idx]  #(B,)
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

            pred_binary = (scores >= 0.5).astype(float)

            tp = np.sum((pred_binary == 1) & (targets == 1))
            fp = np.sum((pred_binary == 1) & (targets == 0))
            precision = tp / (tp + fp + 1e-6)
            per_label_precision.append(float(precision))

            fn = np.sum((pred_binary == 0) & (targets == 1))
            recall = tp / (tp + fn + 1e-6)
            per_label_recall.append(float(recall))

            tn = np.sum((pred_binary == 0) & (targets == 0))
            accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-6)
            per_label_accuracy.append(float(accuracy))

    macro_pr = float(np.nanmean(per_label_pr)) if per_label_pr else float("nan")
    macro_precision = float(np.nanmean(per_label_precision)) if per_label_precision else float("nan")
    macro_recall = float(np.nanmean(per_label_recall)) if per_label_recall else float("nan")
    macro_accuracy = float(np.nanmean(per_label_accuracy)) if per_label_accuracy else float("nan")
    avg_loss = total_loss / (total_weight + 1e-6)

    return {
        "loss": avg_loss,
        "macro_pr_auc": macro_pr,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_accuracy": macro_accuracy,
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
    
#hyperparam search
HYPERPARAM_SPACE = {
    "lr": (1e-5, 1e-2),  
    "weight_decay": (1e-6, 1e-2),  
    "batch_size": [32, 64, 128, 256],
    "d_model": [64, 128, 256],
    "n_layers": [2, 3, 4],
    "transformer_dropout": (0.05, 0.3),
    "cnn_channels": [32, 64, 128],
    "fusion_hidden_dim": [128, 256, 512],
    "fusion_dropout": (0.1, 0.4),
    "spatial_sigma": (0.5, 4.0),
    "label_smoothing": (0.0, 0.1),
}

_fixed_bs_env = os.getenv("FIX_BATCH_SIZE")
if _fixed_bs_env:
    try:
        _fixed_bs = int(_fixed_bs_env)
        HYPERPARAM_SPACE["batch_size"] = [_fixed_bs]
        _log(f"[HYPERPARAM] Forcing batch_size={_fixed_bs} via FIX_BATCH_SIZE")
    except ValueError:
        _log(f"[HYPERPARAM] Invalid FIX_BATCH_SIZE='{_fixed_bs_env}', ignoring.")

OPTUNA_STORAGE_PATH = SEARCH_OUTPUT_DIR / "optuna_study.db"

def sample_optuna_config(trial: "optuna.Trial", max_epochs: int = MAX_EPOCHS) -> TrainConfig:
    """
    sample hyperparameters using Optuna's Bayesian optimization
    """
    space = HYPERPARAM_SPACE
    return TrainConfig(
        d_model=trial.suggest_categorical("d_model", space["d_model"]),
        n_layers=trial.suggest_categorical("n_layers", space["n_layers"]),
        transformer_dropout=trial.suggest_float("transformer_dropout", *space["transformer_dropout"]),
        cnn_channels=trial.suggest_categorical("cnn_channels", space["cnn_channels"]),
        fusion_hidden_dim=trial.suggest_categorical("fusion_hidden_dim", space["fusion_hidden_dim"]),
        fusion_dropout=trial.suggest_float("fusion_dropout", *space["fusion_dropout"]),
        spatial_sigma=trial.suggest_float("spatial_sigma", *space["spatial_sigma"]),

        batch_size=trial.suggest_categorical("batch_size", space["batch_size"]),
        lr=trial.suggest_float("lr", *space["lr"], log=True),
        weight_decay=trial.suggest_float("weight_decay", *space["weight_decay"], log=True),
        label_smoothing=trial.suggest_float("label_smoothing", *space["label_smoothing"]),
        max_epochs=max_epochs,

        seed=trial.number * 1000 + RANDOM_SEED,
    )


def resolve_max_seq_len(max_history_setting: str | int | None) -> int | None:
    if max_history_setting is None:
        return MAX_SEQ_LEN
    if isinstance(max_history_setting, str):
        if max_history_setting.lower() == "all":
            return None
        if max_history_setting.lower().startswith("last_"):
            return int(max_history_setting.split("_")[-1])
    return int(max_history_setting)


def create_model_for_config(config: TrainConfig, feature_dim: int, max_seq_len: int, num_labels: int) -> HybridModel:
    return HybridModel(
        sequence_feature_dim=feature_dim,
        d_model=config.d_model,
        n_layers=config.n_layers,
        d_ff_multiplier=config.d_ff_multiplier,
        transformer_dropout=config.transformer_dropout,
        cnn_channels=config.cnn_channels,
        cnn_dropout=config.cnn_dropout,
        spatial_sigma=config.spatial_sigma,
        fusion_hidden_dim=config.fusion_hidden_dim,
        fusion_dropout=config.fusion_dropout,
        max_seq_len=max_seq_len,
        num_labels=num_labels,
    ).to(DEVICE)


def train_single_config(
    run_id: int,
    config: TrainConfig,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    spatial_data: List[Dict],
    feature_cols: List[str],
    label_cols: List[str],
    eligibility_cols: List[str],
    pos_weight: torch.Tensor,
    wandb_group: str | None = None,
    shuffle: bool = True,
):
    set_global_seed(config.seed)

    max_seq_len = resolve_max_seq_len(config.max_history_minutes)
    max_seq_len_actual = max_seq_len or MAX_SEQ_LEN

    train_ds = HybridDataset(train_df, spatial_data, feature_cols, label_cols, eligibility_cols, max_seq_len)
    val_ds = HybridDataset(val_df, spatial_data, feature_cols, label_cols, eligibility_cols, max_seq_len)
    test_ds = HybridDataset(test_df, spatial_data, feature_cols, label_cols, eligibility_cols, max_seq_len)

    max_seq_len_actual = max(train_ds.max_seq_len, val_ds.max_seq_len, test_ds.max_seq_len, MAX_SEQ_LEN)

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=shuffle, collate_fn=collate_hybrid, num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, collate_fn=collate_hybrid, num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False, collate_fn=collate_hybrid, num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available())

    wandb_run = init_wandb_run(
        name=f"hybrid-run-{run_id:03d}",
        job_type="hyperparameter-search",
        tags=["hybrid", "optuna-eval"],
        group=wandb_group,
        config={**asdict(config), "run_id": run_id, "feature_dim": len(feature_cols), "num_labels": len(label_cols)},
    )

    model = create_model_for_config(config, len(feature_cols), max_seq_len_actual, num_labels=len(label_cols))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    total_steps = max(1, config.max_epochs * len(train_loader))
    scheduler = create_scheduler(optimizer, total_steps)

    best_state = None
    best_val_metric = -float("inf")
    best_epoch = -1
    best_val_metrics = None
    last_val_metrics = None
    patience_counter = 0
    history = []

    for epoch in range(1, config.max_epochs + 1):
        model.train()
        epoch_train_losses = []

        for batch in train_loader:
            seq_inputs = batch["sequence_inputs"].to(DEVICE)
            labels_full = batch["labels"].to(DEVICE)  #(B, T, num_labels)
            eligibility_full = batch["eligibility"].to(DEVICE)  #(B, T, num_labels)
            mask = batch["mask"].to(DEVICE)
            valid_lengths = batch["valid_lengths"].to(DEVICE)  #(B,)

            spatial_data_batch = {
                "player_trails": tuple(t.to(DEVICE) for t in batch["spatial_data"]["player_trails"]),
                "dead_towers": batch["spatial_data"]["dead_towers"].to(DEVICE),
                "obj_status": batch["spatial_data"]["obj_status"].to(DEVICE),
                "kill_ctx": tuple(t.to(DEVICE) for t in batch["spatial_data"]["kill_ctx"]),
            }

            optimizer.zero_grad(set_to_none=True)

            logits = model(seq_inputs, spatial_data_batch, src_key_padding_mask=(mask == 0), valid_lengths=valid_lengths)

            B = seq_inputs.shape[0]
            last_indices = (valid_lengths - 1).clamp(min=0)  #(B,)
            batch_indices = torch.arange(B, device=DEVICE)
            labels = labels_full[batch_indices, last_indices]  #(B, num_labels)
            eligibility = eligibility_full[batch_indices, last_indices]  #(B, num_labels)

            valid_mask = eligibility

            loss = compute_loss(logits, labels, valid_mask, pos_weight, config.label_smoothing)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            epoch_train_losses.append(loss.item())

        val_metrics = evaluate_model(model, val_loader, pos_weight, len(label_cols), DEVICE, label_smoothing=config.label_smoothing)
        last_val_metrics = val_metrics.copy()
        history.append({"epoch": epoch, "val_macro_pr_auc": val_metrics["macro_pr_auc"], "val_loss": val_metrics["loss"]})
        metric = val_metrics["macro_pr_auc"]
        _log(f"[Run {run_id}] Epoch {epoch}/{config.max_epochs} | Train loss={float(np.mean(epoch_train_losses)) if epoch_train_losses else float('nan'):.4f} | Val loss={val_metrics['loss']:.4f} | Val PR-AUC={val_metrics['macro_pr_auc']:.4f}")

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
                if wandb_run:
                    wandb_run.log({"early_stopping/epoch": epoch, "early_stopping/patience_counter": patience_counter})
                break

        if wandb_run:
            horizon_avgs = compute_horizon_avg_pr_auc(val_metrics["per_label_pr_auc"])

            log_payload = {
                "epoch": epoch,
                "train/loss": float(np.mean(epoch_train_losses)) if epoch_train_losses else float("nan"),
                "val/loss": val_metrics["loss"],
                "val/macro_pr_auc": val_metrics["macro_pr_auc"],
                "val/avg_pr_auc_1min": horizon_avgs["avg_pr_auc_1min"],
                "val/avg_pr_auc_2min": horizon_avgs["avg_pr_auc_2min"],
                "val/avg_pr_auc_3min": horizon_avgs["avg_pr_auc_3min"],
                "val/macro_precision": val_metrics["macro_precision"],
                "val/macro_recall": val_metrics["macro_recall"],
                "val/macro_accuracy": val_metrics["macro_accuracy"],
                "train/learning_rate": scheduler.get_last_lr()[0],
            }
            for idx, label_name in enumerate(label_cols):
                pretty = format_label_for_logging(label_name)
                pr_auc = val_metrics["per_label_pr_auc"][idx]
                if not np.isnan(pr_auc):
                    log_payload[f"val/pr_auc_{pretty}"] = pr_auc
                precision = val_metrics["per_label_precision"][idx]
                if not np.isnan(precision):
                    log_payload[f"val/precision_{pretty}"] = precision
                recall = val_metrics["per_label_recall"][idx]
                if not np.isnan(recall):
                    log_payload[f"val/recall_{pretty}"] = recall
                accuracy = val_metrics["per_label_accuracy"][idx]
                if not np.isnan(accuracy):
                    log_payload[f"val/accuracy_{pretty}"] = accuracy
            wandb_run.log(log_payload, step=epoch)

            if improved:
                wandb_run.log({"best/val_macro_pr_auc": best_val_metric, "best/epoch": best_epoch}, step=epoch)

    if best_state is None:
        best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        best_epoch = epoch
    if best_val_metrics is None:
        best_val_metrics = last_val_metrics or {}

    model.load_state_dict(best_state)
    test_metrics = evaluate_model(model, test_loader, pos_weight, len(label_cols), DEVICE, label_smoothing=config.label_smoothing)

    run_dir = SEARCH_OUTPUT_DIR / f"run_{run_id:03d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    model_path = run_dir / "best_model.pt"
    torch.save({
        "config": asdict(config),
        "state_dict": best_state,
        "val_macro_pr_auc": best_val_metric,
        "test_metrics": test_metrics,
    }, model_path)

    summary_path = run_dir / "summary.json"
    with open(summary_path, "w") as fp:
        json.dump({
            "run_id": run_id,
            "config": asdict(config),
            "best_epoch": best_epoch,
            "val_macro_pr_auc": best_val_metric,
            "best_val_metrics": best_val_metrics,
            "test_metrics": test_metrics,
            "history": history,
        }, fp, indent=2)

    if wandb_run:
        test_horizon_avgs = compute_horizon_avg_pr_auc(test_metrics["per_label_pr_auc"])

        log_payload = {
            "final/val_macro_pr_auc": best_val_metric,
            "final/val_macro_recall": best_val_metrics.get("macro_recall", float("nan")) if isinstance(best_val_metrics, dict) else float("nan"),
            "final/test_macro_pr_auc": test_metrics["macro_pr_auc"],
            "final/test/avg_pr_auc_1min": test_horizon_avgs["avg_pr_auc_1min"],
            "final/test/avg_pr_auc_2min": test_horizon_avgs["avg_pr_auc_2min"],
            "final/test/avg_pr_auc_3min": test_horizon_avgs["avg_pr_auc_3min"],
            "final/test_macro_precision": test_metrics["macro_precision"],
            "final/test_macro_recall": test_metrics["macro_recall"],
            "final/test_macro_accuracy": test_metrics["macro_accuracy"],
            "final/test_loss": test_metrics["loss"],
            "final/best_epoch": best_epoch,
        }
        for idx, label_name in enumerate(label_cols):
            pretty = format_label_for_logging(label_name)
            if not np.isnan(test_metrics["per_label_pr_auc"][idx]):
                log_payload[f"final/test/pr_auc_{pretty}"] = test_metrics["per_label_pr_auc"][idx]
            if not np.isnan(test_metrics["per_label_precision"][idx]):
                log_payload[f"final/test/precision_{pretty}"] = test_metrics["per_label_precision"][idx]
            if not np.isnan(test_metrics["per_label_recall"][idx]):
                log_payload[f"final/test/recall_{pretty}"] = test_metrics["per_label_recall"][idx]
            if not np.isnan(test_metrics["per_label_accuracy"][idx]):
                log_payload[f"final/test/accuracy_{pretty}"] = test_metrics["per_label_accuracy"][idx]
        wandb_run.log(log_payload)
        try:
            artifact = wandb.Artifact(f"hybrid-model-run-{run_id:03d}", type="model")
            artifact.add_file(str(model_path))
            artifact.add_file(str(summary_path))
            wandb_run.log_artifact(artifact)
        except Exception as e:
            _log(f"[WARN] Failed to save wandb artifact: {e}")
        wandb_run.finish()

    return {
        "run_id": run_id,
        "config": config,
        "best_epoch": best_epoch,
        "val_macro_pr_auc": best_val_metric,
        "val_macro_recall": best_val_metrics.get("macro_recall", float("nan")) if isinstance(best_val_metrics, dict) else float("nan"),
        "test_macro_pr_auc": test_metrics["macro_pr_auc"],
        "test_macro_recall": test_metrics["macro_recall"],
        "test_metrics": test_metrics,
    }


# optuna search
def create_optuna_study(study_name: str, storage_path: Path | None = None) -> "optuna.Study":
    """Create or load an Optuna study with SQLite storage for offline use."""
    storage_path = storage_path or OPTUNA_STORAGE_PATH
    storage_url = f"sqlite:///{storage_path}?timeout=300"

    storage_path.parent.mkdir(parents=True, exist_ok=True)

    sampler = TPESampler(seed=HYPERPARAM_SEARCH_SEED, multivariate=True)

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        sampler=sampler,
        direction="maximize",  #maximize PR-AUC
        load_if_exists=True,  
    )
    return study

def optuna_objective(
    trial: "optuna.Trial",
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    spatial_data: List[Dict],
    feature_cols: List[str],
    label_cols: List[str],
    eligibility_cols: List[str],
    pos_weight: torch.Tensor,
    max_epochs: int = MAX_EPOCHS,
    wandb_group: str | None = None,
    shuffle: bool = True,
) -> float:
    """
    optuna objective function for a single trial
    """
    config = sample_optuna_config(trial, max_epochs=max_epochs)
    set_global_seed(config.seed)

    max_seq_len = resolve_max_seq_len(config.max_history_minutes)

    train_ds = HybridDataset(train_df, spatial_data, feature_cols, label_cols, eligibility_cols, max_seq_len)
    val_ds = HybridDataset(val_df, spatial_data, feature_cols, label_cols, eligibility_cols, max_seq_len)

    max_seq_len_actual = max(train_ds.max_seq_len, val_ds.max_seq_len, MAX_SEQ_LEN)

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=shuffle, collate_fn=collate_hybrid, num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, collate_fn=collate_hybrid, num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available())

    wandb_run = init_wandb_run(
        name=f"optuna-trial-{trial.number:03d}",
        job_type="optuna-trial",
        tags=["hybrid", "optuna", "bayesian"],
        group=wandb_group,
        config={**asdict(config), "trial_number": trial.number, "feature_dim": len(feature_cols)},
    )

    model = create_model_for_config(config, len(feature_cols), max_seq_len_actual, num_labels=len(label_cols))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    total_steps = max(1, config.max_epochs * len(train_loader))
    scheduler = create_scheduler(optimizer, total_steps)

    best_val_metric = -float("inf")
    best_val_metrics = None
    patience_counter = 0

    for epoch in range(1, config.max_epochs + 1):
        model.train()
        epoch_losses = []

        for batch in train_loader:
            seq_inputs = batch["sequence_inputs"].to(DEVICE)
            labels_full = batch["labels"].to(DEVICE)
            eligibility_full = batch["eligibility"].to(DEVICE)
            mask = batch["mask"].to(DEVICE)
            valid_lengths = batch["valid_lengths"].to(DEVICE)

            spatial_data_batch = {
                "player_trails": tuple(t.to(DEVICE) for t in batch["spatial_data"]["player_trails"]),
                "dead_towers": batch["spatial_data"]["dead_towers"].to(DEVICE),
                "obj_status": batch["spatial_data"]["obj_status"].to(DEVICE),
                "kill_ctx": tuple(t.to(DEVICE) for t in batch["spatial_data"]["kill_ctx"]),
            }

            optimizer.zero_grad(set_to_none=True)
            logits = model(seq_inputs, spatial_data_batch, src_key_padding_mask=(mask == 0), valid_lengths=valid_lengths)

            B = seq_inputs.shape[0]
            last_indices = (valid_lengths - 1).clamp(min=0)
            batch_indices = torch.arange(B, device=DEVICE)
            labels = labels_full[batch_indices, last_indices]
            eligibility = eligibility_full[batch_indices, last_indices]

            loss = compute_loss(logits, labels, eligibility, pos_weight, config.label_smoothing)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            epoch_losses.append(loss.item())

        val_metrics = evaluate_model(model, val_loader, pos_weight, len(label_cols), DEVICE, label_smoothing=config.label_smoothing)
        metric = val_metrics["macro_pr_auc"]

        if wandb_run:
            horizon_avgs = compute_horizon_avg_pr_auc(val_metrics["per_label_pr_auc"])
            wandb_run.log({
                "epoch": epoch,
                "train/loss": float(np.mean(epoch_losses)),
                "val/loss": val_metrics["loss"],
                "val/macro_pr_auc": metric,
                "val/avg_pr_auc_1min": horizon_avgs["avg_pr_auc_1min"],
                "val/avg_pr_auc_2min": horizon_avgs["avg_pr_auc_2min"],
                "val/avg_pr_auc_3min": horizon_avgs["avg_pr_auc_3min"],
                "val/macro_precision": val_metrics["macro_precision"],
                "val/macro_recall": val_metrics["macro_recall"],
                "val/macro_accuracy": val_metrics["macro_accuracy"],
            }, step=epoch)

        trial.report(metric, epoch)

        if trial.should_prune():
            if wandb_run:
                wandb_run.log({"pruned": True, "pruned_at_epoch": epoch})
                wandb_run.finish()
            raise optuna.TrialPruned()

        if metric > best_val_metric + 1e-4:
            best_val_metric = metric
            best_val_metrics = val_metrics.copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                break

    if wandb_run:
        final_metrics = best_val_metrics if best_val_metrics is not None else val_metrics
        wandb_run.log({
            "final/val_macro_pr_auc": best_val_metric,
            "final/val_macro_precision": final_metrics.get("macro_precision", float("nan")),
            "final/val_macro_recall": final_metrics.get("macro_recall", float("nan")),
            "final/val_macro_accuracy": final_metrics.get("macro_accuracy", float("nan")),
        })
        wandb_run.finish()

    return best_val_metric

def run_optuna_search(
    num_trials: int,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    spatial_data: List[Dict],
    feature_cols: List[str],
    label_cols: List[str],
    eligibility_cols: List[str],
    pos_weight: torch.Tensor,
    study_name: str = "hybrid_model_search",
    storage_path: Path | None = None,
    max_epochs: int = MAX_EPOCHS,
    wandb_group: str | None = None,
    trial_id: int | None = None,
    shuffle: bool = True,
):
    """
    run Optuna Bayesian hyperparameter search
    """
    study = create_optuna_study(study_name, storage_path)
    wandb_group = wandb_group or f"optuna-{study_name}"

    def objective(trial):
        return optuna_objective(
            trial, train_df, val_df, spatial_data,
            feature_cols, label_cols, eligibility_cols, pos_weight,
            max_epochs=max_epochs, wandb_group=wandb_group, shuffle=shuffle
        )

    if trial_id is not None:
        _log(f"Running Optuna trial {trial_id} (study: {study_name})")

        n_existing = len(study.trials)
        if trial_id >= n_existing:
            n_to_run = trial_id - n_existing + 1
            study.optimize(objective, n_trials=n_to_run, show_progress_bar=False)
        else:
            existing_trial = study.trials[trial_id]
            if existing_trial.state == optuna.trial.TrialState.COMPLETE:
                completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
                if completed_trials:
                    _log(f"Trial {trial_id} already completed. Best value so far: {study.best_value:.4f}")
                else:
                    _log(f"Trial {trial_id} already completed.")
            elif existing_trial.state == optuna.trial.TrialState.RUNNING:
                _log(f"Trial {trial_id} is currently running by another job. Waiting...")
                import time
                max_wait = 300  
                waited = 0
                while waited < max_wait:
                    time.sleep(10)
                    waited += 10
                    study = create_optuna_study(study_name, storage_path)
                    if trial_id < len(study.trials):
                        current_state = study.trials[trial_id].state
                        if current_state != optuna.trial.TrialState.RUNNING:
                            break
                    else:
                        break

                study = create_optuna_study(study_name, storage_path)
                if trial_id < len(study.trials):
                    if study.trials[trial_id].state == optuna.trial.TrialState.COMPLETE:
                        _log(f"Trial {trial_id} completed by another job.")
                    else:
                        _log(f"Trial {trial_id} is now in state {study.trials[trial_id].state} after {waited}s.")
                else:
                    _log(f"Trial {trial_id} no longer exists in study after {waited}s.")
            else:
                _log(f"Trial {trial_id} exists in state {existing_trial.state}. Running new trial")
                study.optimize(objective, n_trials=1, show_progress_bar=False)
    else:
        _log(f"Starting Optuna Bayesian search with {num_trials} total trials (study: {study_name})")
        _log(f"Storage: {storage_path or OPTUNA_STORAGE_PATH}")
        _log("Running in queue mode: will run trials sequentially until all are complete")

        optuna.logging.set_verbosity(optuna.logging.INFO)

        import time
        while True:
            study = create_optuna_study(study_name, storage_path)
            completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])

            if completed >= num_trials:
                _log(f"All {num_trials} trials completed! (Found {completed} completed trials)")
                break

            _log(f"Progress: {completed}/{num_trials} trials completed. Running next trial")

            study.optimize(
                objective,
                n_trials=1,
                show_progress_bar=False,
                gc_after_trial=True,
            )

            time.sleep(1)

    _log(f"\n{'='*60}")
    _log("Optuna Search Complete")
    _log(f"{'='*60}")
    _log(f"Total trials: {len(study.trials)}")

    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if completed_trials:
        _log(f"Best trial: {study.best_trial.number}")
        _log(f"Best validation macro PR-AUC: {study.best_value:.4f}")
        _log("\nBest hyperparameters:")
        for key, value in study.best_params.items():
            _log(f"  {key}: {value}")

        best_config = TrainConfig(**{
            k: v for k, v in study.best_params.items()
            if k in TrainConfig.__dataclass_fields__
        })
        best_config.seed = study.best_trial.number * 1000 + RANDOM_SEED
    else:
        _log("No completed trials yet.")
        best_config = None

    summary_path = (storage_path or OPTUNA_STORAGE_PATH).parent / "optuna_summary.json"
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if completed_trials:
        with open(summary_path, "w") as f:
            json.dump({
                "study_name": study_name,
                "best_trial": study.best_trial.number,
                "best_value": study.best_value,
                "best_params": study.best_params,
                "n_trials": len(study.trials),
            }, f, indent=2)
    else:
        with open(summary_path, "w") as f:
            json.dump({
                "study_name": study_name,
                "best_trial": None,
                "best_value": None,
                "best_params": None,
                "n_trials": len(study.trials),
            }, f, indent=2)
    _log(f"\nSummary saved to {summary_path}")

    if test_df is not None and len(test_df) > 0 and best_config is not None:
        storage_dir = (storage_path or OPTUNA_STORAGE_PATH).parent
        lock_file_path = storage_dir / f"{study_name}_best_eval.lock"
        eval_done_file = storage_dir / f"{study_name}_best_eval.done"

        if eval_done_file.exists():
            _log(f"\nBest model evaluation already completed (found {eval_done_file}). Skipping.")
        else:
            lock_file = None
            try:
                lock_file = open(lock_file_path, 'w')
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

                if eval_done_file.exists():
                    _log(f"\nBest model evaluation already completed by another job. Skipping.")
                else:
                    _log("\nAcquired lock for best model evaluation. Evaluating best config on test set")
                    result = train_single_config(
                        run_id=study.best_trial.number,
                        config=best_config,
                        train_df=train_df,
                        val_df=val_df,
                        test_df=test_df,
                        spatial_data=spatial_data,
                        feature_cols=feature_cols,
                        label_cols=label_cols,
                        eligibility_cols=eligibility_cols,
                        pos_weight=pos_weight,
                        wandb_group=wandb_group + "-final",
                        shuffle=False,  
                    )
                    if result:
                        _log(f"Test macro PR-AUC: {result['test_macro_pr_auc']:.4f}")

                    eval_done_file.touch()
                    _log(f"Best model evaluation complete. Marker file created: {eval_done_file}")
            except (IOError, OSError) as e:
                _log(f"\nCould not acquire lock for best model evaluation (another job is running it): {e}")
                _log("Skipping evaluation - will be handled by the job that acquired the lock.")
            finally:
                if lock_file is not None:
                    try:
                        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                    except:
                        pass
                    lock_file.close()

    return study

# main entry point
def main():
    parser_prelim = argparse.ArgumentParser(add_help=False)
    parser_prelim.add_argument("--mode", type=str, default="train",
                               choices=["train", "optuna", "evaluate", "trial"])
    args_prelim, _ = parser_prelim.parse_known_args()

    if args_prelim.mode in ["optuna"]:
        jitter = random.uniform(0, 15)
        _log(f"Startup jitter: sleeping {jitter:.2f} seconds to prevent thundering herd")
        time.sleep(jitter)

    parser = argparse.ArgumentParser(description="Hybrid Transformer-CNN Model Training", add_help=False)
    parser.add_argument("--mode", type=str, default="train",
                       choices=["train", "optuna", "evaluate", "trial"])
    parser.add_argument("--epochs", type=int, default=MAX_EPOCHS) 
    parser.add_argument("--trials", type=int, default=HYPERPARAM_SEARCH_TRIALS)
    parser.add_argument("--trial-id", type=int)
    parser.add_argument("--seed", type=int, default=HYPERPARAM_SEARCH_SEED)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--spatial-sigma", type=float, default=1.5)
    parser.add_argument("--config-file", type=str) 
    parser.add_argument("--config-index", type=int) 
    parser.add_argument("--config-json", type=str) 
    parser.add_argument("--run-id", type=int) 
    parser.add_argument("--wandb-group", type=str)
    parser.add_argument("--data-limit", type=int) 
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--study-name", type=str, default="hybrid_model_search")
    parser.add_argument("--storage-path", type=str) 
    parser.add_argument("--pruning", action="store_true", default=True)
    parser.add_argument("--no-pruning", action="store_false", dest="pruning")
    args = parser.parse_args()

    if args.debug:
        args.data_limit = 10
        args.epochs = 50
        _log("[DEBUG MODE] Overriding settings: data_limit=10, epochs=50, shuffle=False, early_stopping=disabled")

    _log("Loading data...")

    spatial_data = torch.load(SPATIAL_DATA_PACKED_PATH, weights_only=False)

    _log("Loading sequence data from godview_cleaned.parquet...")
    team_sequence_df = pd.read_parquet("godview_cleaned.parquet")

   
    if args.data_limit:
        match_ids = team_sequence_df["matchId"].unique()[:args.data_limit]
        team_sequence_df = team_sequence_df[team_sequence_df["matchId"].isin(match_ids)]
        _log(f"Limited to {args.data_limit} games.")

    _log(f"Filling NaN values from schema drift (shape before: {team_sequence_df.shape})...")
    team_sequence_df = team_sequence_df.fillna(0)
    _log(f"Shape after fillna: {team_sequence_df.shape}")

    with open(TEAM_SEQUENCE_METADATA_PATH, "r") as meta_file:
        team_sequence_meta = json.load(meta_file)

    team_label_cols = team_sequence_meta["team_label_cols"]
    team_eligibility_cols = team_sequence_meta.get("team_eligibility_cols", [])
    team_sequence_feature_cols = team_sequence_meta["team_sequence_feature_cols"]

    num_spatial_samples = spatial_data.get("num_samples", len(spatial_data["meta"]))
    _log(f"Loaded {num_spatial_samples:,} spatial samples (packed format), {len(team_sequence_df):,} sequence rows")

    if args.data_limit:
        match_ids = team_sequence_df["matchId"].unique()[:args.data_limit]
        team_sequence_df = team_sequence_df[team_sequence_df["matchId"].isin(match_ids)]

        def normalize_match_id(mid):
            mid_str = str(mid)
            if mid_str.endswith('.0'):
                mid_str = mid_str[:-2]
            return mid_str

        match_hashes = {_hash_match_id(normalize_match_id(mid)) for mid in match_ids}
        meta_np = spatial_data["meta"].numpy()
        valid_indices = []
        for i in range(len(meta_np)):
            if int(meta_np[i, 0]) in match_hashes:
                valid_indices.append(i)

        if len(valid_indices) < len(meta_np):
            _log(f"Filtering packed data to {len(valid_indices):,} samples...")
            valid_indices = np.array(valid_indices)
            spatial_data = {
                "meta": spatial_data["meta"][valid_indices],
                "obj_status": spatial_data["obj_status"][valid_indices],
                "towers": spatial_data["towers"][valid_indices],
                "kills_idx": spatial_data["kills_idx"][valid_indices],
                "trails_idx": spatial_data["trails_idx"][valid_indices],
                "kills_val": spatial_data["kills_val"],  # keep all kills, update indices
                "trails_val": spatial_data["trails_val"],  # keep all trails, update indices
                "num_samples": len(valid_indices),
            }
            meta_np = spatial_data["meta"].numpy()
            spatial_lookup = {}
            for i in range(len(meta_np)):
                match_hash = int(meta_np[i, 0])
                minute = int(meta_np[i, 1])
                spatial_lookup[(match_hash, minute)] = i
            spatial_data["spatial_lookup"] = spatial_lookup

        _log(f"Limited to {args.data_limit} games: {len(team_sequence_df):,} rows, {len(valid_indices):,} spatial samples")
    _log(f"Features: {len(team_sequence_feature_cols)}, Labels: {team_label_cols}")

    _log("splitting data...")
    match_ids = team_sequence_df["matchId"].unique()
    train_matches, holdout_matches = train_test_split(
        match_ids, test_size=VAL_SPLIT + TEST_SPLIT, random_state=RANDOM_SEED, shuffle=True
    )
    val_fraction = VAL_SPLIT / (VAL_SPLIT + TEST_SPLIT)
    val_matches, test_matches = train_test_split(
        holdout_matches, test_size=1 - val_fraction, random_state=RANDOM_SEED, shuffle=True
    )

    train_df = team_sequence_df[team_sequence_df["matchId"].isin(train_matches)].reset_index(drop=True)
    val_df = team_sequence_df[team_sequence_df["matchId"].isin(val_matches)].reset_index(drop=True)
    test_df = team_sequence_df[team_sequence_df["matchId"].isin(test_matches)].reset_index(drop=True)

    _log("normalizing features...")
    continuous_cols = _detect_continuous_features(team_sequence_df, team_sequence_feature_cols)
    norm_stats = _compute_normalization_stats(train_df, continuous_cols)
    train_df = _apply_normalization(train_df, norm_stats)
    val_df = _apply_normalization(val_df, norm_stats)
    test_df = _apply_normalization(test_df, norm_stats)

    _log(f"Splits -> train: {len(train_df)} | val: {len(val_df)} | test: {len(test_df)}")

    if team_eligibility_cols:
        pos_weights = [
            _compute_pos_weight(train_df, label_col, elig_col)
            for label_col, elig_col in zip(team_label_cols, team_eligibility_cols)
        ]
    else:
        pos_weights = [
            max(1.0, (len(train_df) / (train_df[label_col].sum() + 1e-6)))
            for label_col in team_label_cols
        ]
    pos_weight = torch.tensor(pos_weights, dtype=torch.float32, device=DEVICE)
    _log(f"Positive weights: {pos_weights}")

    shuffle = not args.debug

    if args.mode == "optuna":
        storage_path = Path(args.storage_path) if args.storage_path else None
        run_optuna_search(
            num_trials=args.trials,
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            spatial_data=spatial_data,
            feature_cols=team_sequence_feature_cols,
            label_cols=team_label_cols,
            eligibility_cols=team_eligibility_cols,
            pos_weight=pos_weight,
            study_name=args.study_name,
            storage_path=storage_path,
            max_epochs=args.epochs,
            wandb_group=args.wandb_group,
            trial_id=args.trial_id,
            shuffle=shuffle,
        )
        return

    if args.mode == "trial":
        if args.config_json:
            payload = json.loads(args.config_json)
        elif args.config_file:
            if args.config_index is None:
                raise ValueError("Provide --config-index with --config-file")
            with open(args.config_file) as fp:
                config_list = json.load(fp)
            payload = config_list[args.config_index]
        else:
            raise ValueError("Provide --config-json or --config-file for trial mode")

        run_id = args.run_id or payload.get("run_id", 0)
        config = TrainConfig(**{k: v for k, v in payload.items() if k != "run_id"})
        _log(f"Running trial {run_id} with config: {config}")

        result = train_single_config(
            run_id, config, train_df, val_df, test_df, spatial_data,
            team_sequence_feature_cols, team_label_cols, team_eligibility_cols, pos_weight,
            wandb_group=args.wandb_group, shuffle=shuffle
        )
        if result:
            _log(f"Val PR-AUC={result['val_macro_pr_auc']:.4f}, Test PR-AUC={result['test_macro_pr_auc']:.4f}")
        return

    _log(f"\nTraining hybrid model on {DEVICE}...")

    patience = args.epochs if args.debug else EARLY_STOPPING_PATIENCE

    config = TrainConfig(
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        spatial_sigma=args.spatial_sigma,
        max_epochs=args.epochs,
        patience=patience,
        seed=args.seed,
    )

    wandb_run = init_wandb_run(
        name="hybrid-single-train",
        job_type="train",
        tags=["hybrid", "single-run"],
        config=asdict(config),
    )

    train_ds = HybridDataset(train_df, spatial_data, team_sequence_feature_cols, team_label_cols, team_eligibility_cols)
    val_ds = HybridDataset(val_df, spatial_data, team_sequence_feature_cols, team_label_cols, team_eligibility_cols)
    test_ds = HybridDataset(test_df, spatial_data, team_sequence_feature_cols, team_label_cols, team_eligibility_cols)

    max_seq_len = max(train_ds.max_seq_len, val_ds.max_seq_len, test_ds.max_seq_len, MAX_SEQ_LEN)

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=shuffle, collate_fn=collate_hybrid, num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, collate_fn=collate_hybrid, num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False, collate_fn=collate_hybrid, num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available())

    _log(f"Train: {len(train_ds)} samples, Val: {len(val_ds)} samples, Test: {len(test_ds)} samples")

    model = create_model_for_config(config, len(team_sequence_feature_cols), max_seq_len, num_labels=len(team_label_cols))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    total_steps = config.max_epochs * len(train_loader)
    scheduler = create_scheduler(optimizer, total_steps)

    _log(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    best_val_metric = -float("inf")
    best_state = None
    best_epoch = 0
    patience_counter = 0

    for epoch in range(1, config.max_epochs + 1):
        model.train()
        train_losses = []

        for batch in train_loader:
            seq_inputs = batch["sequence_inputs"].to(DEVICE)
            labels_full = batch["labels"].to(DEVICE)  #(B, T, num_labels)
            eligibility_full = batch["eligibility"].to(DEVICE)  #(B, T, num_labels)
            mask = batch["mask"].to(DEVICE)
            valid_lengths = batch["valid_lengths"].to(DEVICE)  #(B,)

            spatial_data_batch = {
                "player_trails": tuple(t.to(DEVICE) for t in batch["spatial_data"]["player_trails"]),
                "dead_towers": batch["spatial_data"]["dead_towers"].to(DEVICE),
                "obj_status": batch["spatial_data"]["obj_status"].to(DEVICE),
                "kill_ctx": tuple(t.to(DEVICE) for t in batch["spatial_data"]["kill_ctx"]),
            }

            optimizer.zero_grad(set_to_none=True)

            logits = model(seq_inputs, spatial_data_batch, src_key_padding_mask=(mask == 0), valid_lengths=valid_lengths)

            B = seq_inputs.shape[0]
            last_indices = (valid_lengths - 1).clamp(min=0)  #(B,)
            batch_indices = torch.arange(B, device=DEVICE)
            labels = labels_full[batch_indices, last_indices]  #(B, num_labels)
            eligibility = eligibility_full[batch_indices, last_indices]  #(B, num_labels)

            valid_mask = eligibility

            loss = compute_loss(logits, labels, valid_mask, pos_weight, config.label_smoothing)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            train_losses.append(loss.item())

        val_metrics = evaluate_model(model, val_loader, pos_weight, len(team_label_cols), DEVICE, label_smoothing=config.label_smoothing)

        train_loss_mean = np.mean(train_losses)
        _log(f"Epoch {epoch}/{config.max_epochs} | Train loss: {train_loss_mean:.4f} | "
              f"Val loss: {val_metrics['loss']:.4f} | Val PR-AUC: {val_metrics['macro_pr_auc']:.4f}")

        if wandb_run:
            horizon_avgs = compute_horizon_avg_pr_auc(val_metrics["per_label_pr_auc"])
            log_data = {
                "epoch": epoch,
                "train/loss": train_loss_mean,
                "val/loss": val_metrics["loss"],
                "val/macro_pr_auc": val_metrics["macro_pr_auc"],
                "val/avg_pr_auc_1min": horizon_avgs["avg_pr_auc_1min"],
                "val/avg_pr_auc_2min": horizon_avgs["avg_pr_auc_2min"],
                "val/avg_pr_auc_3min": horizon_avgs["avg_pr_auc_3min"],
                "val/macro_precision": val_metrics["macro_precision"],
                "val/macro_recall": val_metrics["macro_recall"],
                "val/macro_accuracy": val_metrics["macro_accuracy"],
                "train/lr": scheduler.get_last_lr()[0],
            }
            for i, label in enumerate(team_label_cols):
                pretty = format_label_for_logging(label)
                if not np.isnan(val_metrics["per_label_pr_auc"][i]):
                    log_data[f"val/pr_auc_{pretty}"] = val_metrics["per_label_pr_auc"][i]
                if not np.isnan(val_metrics["per_label_precision"][i]):
                    log_data[f"val/precision_{pretty}"] = val_metrics["per_label_precision"][i]
                if not np.isnan(val_metrics["per_label_recall"][i]):
                    log_data[f"val/recall_{pretty}"] = val_metrics["per_label_recall"][i]
                if not np.isnan(val_metrics["per_label_accuracy"][i]):
                    log_data[f"val/accuracy_{pretty}"] = val_metrics["per_label_accuracy"][i]
            wandb_run.log(log_data, step=epoch)

        if val_metrics["macro_pr_auc"] > best_val_metric + 1e-4:
            best_val_metric = val_metrics["macro_pr_auc"]
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                _log(f"Early stopping at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics = evaluate_model(model, test_loader, pos_weight, len(team_label_cols), DEVICE, label_smoothing=config.label_smoothing)
    _log(f"\nTest Results (best epoch {best_epoch}):")
    _log(f"  Loss: {test_metrics['loss']:.4f}")
    _log(f"  Macro PR-AUC: {test_metrics['macro_pr_auc']:.4f}")
    for i, label in enumerate(team_label_cols):
        _log(f"  {label}: PR-AUC={test_metrics['per_label_pr_auc'][i]:.4f}")

    if wandb_run:
        test_horizon_avgs = compute_horizon_avg_pr_auc(test_metrics["per_label_pr_auc"])
        log_data = {
            "test/loss": test_metrics["loss"],
            "test/macro_pr_auc": test_metrics["macro_pr_auc"],
            "test/avg_pr_auc_1min": test_horizon_avgs["avg_pr_auc_1min"],
            "test/avg_pr_auc_2min": test_horizon_avgs["avg_pr_auc_2min"],
            "test/avg_pr_auc_3min": test_horizon_avgs["avg_pr_auc_3min"],
            "test/macro_precision": test_metrics["macro_precision"],
            "test/macro_recall": test_metrics["macro_recall"],
            "test/macro_accuracy": test_metrics["macro_accuracy"],
        }
        for i, label in enumerate(team_label_cols):
            pretty = format_label_for_logging(label)
            if not np.isnan(test_metrics["per_label_pr_auc"][i]):
                log_data[f"test/pr_auc_{pretty}"] = test_metrics["per_label_pr_auc"][i]
            if not np.isnan(test_metrics["per_label_precision"][i]):
                log_data[f"test/precision_{pretty}"] = test_metrics["per_label_precision"][i]
            if not np.isnan(test_metrics["per_label_recall"][i]):
                log_data[f"test/recall_{pretty}"] = test_metrics["per_label_recall"][i]
            if not np.isnan(test_metrics["per_label_accuracy"][i]):
                log_data[f"test/accuracy_{pretty}"] = test_metrics["per_label_accuracy"][i]
        wandb_run.log(log_data, step=config.max_epochs + 1)
        wandb_run.finish()

    save_path = SEARCH_OUTPUT_DIR / "single_train_model.pt"
    torch.save({
        "config": asdict(config),
        "state_dict": best_state or model.state_dict(),
        "val_macro_pr_auc": best_val_metric,
        "test_metrics": test_metrics,
        "norm_stats": norm_stats,
    }, save_path)
    _log(f"\nModel saved to {save_path}")


if __name__ == "__main__":
    main()

