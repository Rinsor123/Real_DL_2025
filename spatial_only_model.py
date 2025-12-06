"""
Spatial-Only Model for Objective Prediction
Uses only spatial data 
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

#WandB setup
WANDB_AVAILABLE = False
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    pass

USE_WANDB = os.getenv("USE_WANDB", "1").lower() not in {"0", "false", "off", "no"}
WANDB_MODE = os.getenv("WANDB_MODE", "offline")
WANDB_PROJECT = os.getenv("WANDB_PROJECT", "spatial-only-model")
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
SPATIAL_DATA_PATH = DATA_ROOT / "spatial_data_raw.pt"
SPATIAL_DATA_PACKED_PATH = DATA_ROOT / "spatial_data_packed.pt"
TEAM_SEQUENCE_FEATURES_PATH = DATA_ROOT / "team_sequence_dataset"
TEAM_SEQUENCE_METADATA_PATH = DATA_ROOT / "team_sequence_metadata.json"

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

BATCH_SIZE = 64
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

#spatial constants
MAP_SIZE = 15000
GRID_SIZE = 64

#tower locations
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

OUTPUT_DIR = DATA_ROOT / "spatial_only_model"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _hash_match_id(match_id: str) -> int:
    """
    Convert matchId string to int32
    """
    h = 5381
    for char in match_id:
        h = ((h << 5) + h) + ord(char)
        h = h & 0xFFFFFFFF
    return h & 0x7FFFFFFF

@dataclass
class TrainConfig:
    cnn_channels: int = 64
    cnn_dropout: float = 0.1
    spatial_sigma: float = 1.5
    hidden_dim: int = 256
    hidden_dropout: float = 0.2
    batch_size: int = BATCH_SIZE
    lr: float = 1e-3
    weight_decay: float = WEIGHT_DECAY
    max_epochs: int = MAX_EPOCHS
    patience: int = EARLY_STOPPING_PATIENCE
    label_smoothing: float = 0.0
    seed: int = RANDOM_SEED

class SpatialInputLayer(nn.Module):
    """
    makes raw game state into multi-channel heatmap
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
        B = dead_towers_mask.shape[0]
        H, W = self.grid_size, self.grid_size
        
        canvas = torch.zeros(B, self.num_channels, H, W, device=dead_towers_mask.device)
        
        #towers
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

        #objectives
        drag_pos = self._to_grid(torch.tensor([10000., 5000.], device=dead_towers_mask.device))
        baron_pos = self._to_grid(torch.tensor([5000., 10000.], device=dead_towers_mask.device))
        has_drag = obj_status[:, 0] > 0.5
        has_baron = obj_status[:, 1] > 0.5
        canvas[has_drag, 3, drag_pos[1], drag_pos[0]] = 1.0
        canvas[has_baron, 3, baron_pos[1], baron_pos[0]] = 1.0

        #kills
        k_coords, k_vals, k_teams, k_b_idx = kill_ctx
        if k_coords.shape[0] > 0:
            k_grid = self._to_grid(k_coords)
            channels = 4 + k_teams.long()
            indices = (k_b_idx, channels, k_grid[:, 1], k_grid[:, 0])
            canvas.index_put_(indices, k_vals, accumulate=True)

        #players
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
    """
    ResNet-style CNN for spatial heatmap processing
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
        #make spatial heatmap
        x = self.renderer(player_trails, dead_towers, kill_ctx, obj_status)
        
        #CNN forward
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

class SpatialOnlyModel(nn.Module):
    """
    Spatial-only model (no sequence features)
    """
    def __init__(
        self,
        cnn_channels: int = 64,
        cnn_dropout: float = 0.1,
        spatial_sigma: float = 1.5,
        hidden_dim: int = 256,
        hidden_dropout: float = 0.2,
        num_labels: int = 2,
    ):
        super().__init__()
        
        #CNN branch
        self.cnn = SpatialCNN(
            base_channels=cnn_channels,
            dropout=cnn_dropout,
            spatial_sigma=spatial_sigma,
        )
        
        #prediction heads 
        assert num_labels == 6, "Expected 6 labels: 3 horizons × 2 objectives"
        cnn_embed_dim = cnn_channels * 8
        
        self.shared_mlp = nn.Sequential(
            nn.Linear(cnn_embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(hidden_dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(hidden_dropout),
        )
        
        self.head_1min = nn.Linear(hidden_dim // 2, 2)  #dragon_1min, baron_1min
        self.head_2min = nn.Linear(hidden_dim // 2, 2)  #dragon_2min, baron_2min
        self.head_3min = nn.Linear(hidden_dim // 2, 2)  #dragon_3min, baron_3min
        
        self.num_labels = num_labels

    def forward(self, spatial_data: Dict,):
        #process spatial data
        cnn_out = self.cnn(
            spatial_data["player_trails"],
            spatial_data["dead_towers"],
            spatial_data["kill_ctx"],
            spatial_data["obj_status"],
        )  #(B, cnn_embed_dim)
        
        #shared feature extraction
        shared_features = self.shared_mlp(cnn_out)  #(B, hidden_dim // 2)
        
        #separate heads for each horizon
        logits_1min = self.head_1min(shared_features)  #(B, 2)
        logits_2min = self.head_2min(shared_features)  #(B, 2)
        logits_3min = self.head_3min(shared_features)  #(B, 2)
        
        #Concatenate in label order: [dragon_1min, dragon_2min, dragon_3min, baron_1min, baron_2min, baron_3min]
        logits = torch.cat([
            logits_1min[:, 0:1],  #dragon_1min
            logits_2min[:, 0:1],  #dragon_2min
            logits_3min[:, 0:1],  #dragon_3min
            logits_1min[:, 1:2],  #baron_1min
            logits_2min[:, 1:2],  #baron_2min
            logits_3min[:, 1:2],  #baron_3min
        ], dim=-1)  #(B, 6)
        
        return logits


class SpatialOnlyDataset(Dataset):
    """
    dataset for spatial-only model 
    """
    def __init__(
        self,
        sequence_df: pd.DataFrame,
        spatial_data: List[Dict] | Dict[str, torch.Tensor],
        label_cols: List[str],
        eligibility_cols: List[str] | None = None,
    ):
        if isinstance(spatial_data, dict) and "meta" in spatial_data:
            self.use_packed = True
            self.packed_data = spatial_data
            self.meta = spatial_data["meta"]
            self.obj_status = spatial_data["obj_status"]
            self.towers = spatial_data["towers"]
            self.kills_val = spatial_data["kills_val"]
            self.kills_idx = spatial_data["kills_idx"]
            self.trails_val = spatial_data["trails_val"]
            self.trails_idx = spatial_data["trails_idx"]
            
            meta_np = self.meta.numpy()
            self.spatial_lookup = {}
            for i in range(len(meta_np)):
                match_hash = int(meta_np[i, 0])
                minute = int(meta_np[i, 1])
                self.spatial_lookup[(match_hash, minute)] = i
        else:
            self.use_packed = False
            self.spatial_lookup = {}
            for state in spatial_data:
                key = (state["matchId"], state["minute"])
                self.spatial_lookup[key] = state
        
        #build samples from sequence_df 
        samples = []
        eligibility_cols = eligibility_cols or []
        
        grouped = sequence_df.sort_values(["matchId", "teamId", "minute"])
        for (match_id, team_id), group in grouped.groupby(["matchId", "teamId"], sort=False):
            labels = group[label_cols].to_numpy(dtype=np.float32)
            minutes = group["minute"].to_numpy()
            labels = np.nan_to_num(labels, nan=0.0, posinf=0.0, neginf=0.0)
            
            if eligibility_cols:
                elig = group[eligibility_cols].to_numpy(dtype=np.float32)
                elig = np.nan_to_num(elig, nan=0.0, posinf=0.0, neginf=0.0)
            else:
                elig = np.ones_like(labels, dtype=np.float32)
            
            #create a sample for each minute 
            for idx, minute in enumerate(minutes):
                if self.use_packed:
                    match_hash = _hash_match_id(match_id)
                    spatial_key = (match_hash, int(minute))
                else:
                    spatial_key = (match_id, int(minute))
                
                if spatial_key in self.spatial_lookup:
                    samples.append({
                        "match_id": match_id,
                        "team_id": team_id,
                        "minute": int(minute),
                        "labels": labels[idx],
                        "eligibility": elig[idx],
                        "spatial_key": spatial_key,
                    })
        
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        item = self.samples[idx]
        team_id = item["team_id"]
        
        if self.use_packed:
            spatial_idx = self.spatial_lookup[item["spatial_key"]]
            spatial_dict = self._process_spatial_state_packed(spatial_idx, team_id)
        else:
            spatial_state = self.spatial_lookup[item["spatial_key"]]
            spatial_dict = self._process_spatial_state(spatial_state, team_id)
        
        return {
            "labels": torch.tensor(item["labels"], dtype=torch.float32),
            "eligibility": torch.tensor(item["eligibility"], dtype=torch.float32),
            **spatial_dict,
        }
    
    def _process_spatial_state_packed(self, idx: int, team_id: int) -> Dict:
        """
        Process spatial state from packed tensors by index
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
        
        for player_idx in range(10):
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
        }
    
    def _process_spatial_state(self, state: Dict, team_id: int) -> Dict:
        """
        Process spatial state dict into tensors (legacy format)
        """
        dead_towers = torch.zeros(30, dtype=torch.float32)
        if state.get('dead_towers'):
            dead_towers[torch.tensor(state['dead_towers'], dtype=torch.long)] = 1.0
        
        obj_status = torch.tensor([
            float(state['objectives']['dragon']),
            float(state['objectives']['baron'])
        ], dtype=torch.float32)
        
        kills_data = state.get('kills', [])
        if not kills_data:
            kills = torch.empty((0, 4), dtype=torch.float32)
        else:
            kills = torch.tensor(kills_data, dtype=torch.float32)
        
        p_coords_list = []
        p_vals_list = []
        p_channels_list = []
        
        trails = state.get('player_trails', {})
        for pid_str, points in trails.items():
            pid = int(pid_str)
            channel = pid - 1
            for pt in points:
                p_coords_list.append([pt[0], pt[1]])
                p_vals_list.append(pt[2])
                p_channels_list.append(channel)
        
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
        }


def collate_spatial_only(batch):
    labels = torch.stack([item["labels"] for item in batch])
    eligibility = torch.stack([item["eligibility"] for item in batch])
    
    #pack spatial data
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
        "labels": labels,
        "eligibility": eligibility,
        "spatial_data": {
            "player_trails": (p_coords, p_vals, p_b_idx, p_channels),
            "dead_towers": dead_towers,
            "obj_status": obj_status,
            "kill_ctx": (k_coords, k_vals, k_teams, k_b_idx),
        },
    }


#Training Functions
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
            labels = batch["labels"].to(device)
            eligibility = batch["eligibility"].to(device)
            
            spatial_data = {
                "player_trails": tuple(t.to(device) for t in batch["spatial_data"]["player_trails"]),
                "dead_towers": batch["spatial_data"]["dead_towers"].to(device),
                "obj_status": batch["spatial_data"]["obj_status"].to(device),
                "kill_ctx": tuple(t.to(device) for t in batch["spatial_data"]["kill_ctx"]),
            }
            
            logits = model(spatial_data)
            
            logits = torch.clamp(logits, min=-100, max=100)
            logits = torch.where(torch.isnan(logits), torch.zeros_like(logits), logits)
            
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
    spatial_data: List[Dict] | Dict[str, torch.Tensor],
    label_cols: List[str],
    eligibility_cols: List[str],
    pos_weight: torch.Tensor,
):
    set_global_seed(config.seed)
    
    train_ds = SpatialOnlyDataset(train_df, spatial_data, label_cols, eligibility_cols)
    val_ds = SpatialOnlyDataset(val_df, spatial_data, label_cols, eligibility_cols)
    test_ds = SpatialOnlyDataset(test_df, spatial_data, label_cols, eligibility_cols)
    
    if len(train_ds) == 0 or len(val_ds) == 0:
        print("Not enough data for training/validation. Skipping.")
        return None
    
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, 
                             collate_fn=collate_spatial_only, num_workers=NUM_WORKERS, 
                             pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, 
                           collate_fn=collate_spatial_only, num_workers=NUM_WORKERS, 
                           pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False, 
                            collate_fn=collate_spatial_only, num_workers=NUM_WORKERS, 
                            pin_memory=torch.cuda.is_available())
    
    wandb_run = init_wandb_run(
        name="spatial-only-run",
        job_type="train",
        tags=["spatial-only"],
        config={**asdict(config), "num_labels": len(label_cols)},
    )
    
    model = SpatialOnlyModel(
        cnn_channels=config.cnn_channels,
        cnn_dropout=config.cnn_dropout,
        spatial_sigma=config.spatial_sigma,
        hidden_dim=config.hidden_dim,
        hidden_dropout=config.hidden_dropout,
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
            labels = batch["labels"].to(DEVICE)
            eligibility = batch["eligibility"].to(DEVICE)
            
            spatial_data_batch = {
                "player_trails": tuple(t.to(DEVICE) for t in batch["spatial_data"]["player_trails"]),
                "dead_towers": batch["spatial_data"]["dead_towers"].to(DEVICE),
                "obj_status": batch["spatial_data"]["obj_status"].to(DEVICE),
                "kill_ctx": tuple(t.to(DEVICE) for t in batch["spatial_data"]["kill_ctx"]),
            }
            
            optimizer.zero_grad(set_to_none=True)
            
            logits = model(spatial_data_batch)
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
    parser = argparse.ArgumentParser(description="Spatial-Only Model Training")
    parser.add_argument("--epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE) 
    parser.add_argument("--lr", type=float, default=1e-3) 
    parser.add_argument("--cnn-channels", type=int, default=64) 
    parser.add_argument("--cnn-dropout", type=float, default=0.1) 
    parser.add_argument("--spatial-sigma", type=float, default=1.5)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--label-smoothing", type=float, default=0.0) 
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--data-limit", type=int) 
    args = parser.parse_args()
    
    print("Loading data...")
    
    #load spatial data
    if SPATIAL_DATA_PACKED_PATH.exists():
        print(f"Loading packed spatial data from {SPATIAL_DATA_PACKED_PATH}")
        spatial_data = torch.load(SPATIAL_DATA_PACKED_PATH, weights_only=False)
        print(f"Loaded packed format with {spatial_data.get('num_samples', 'unknown')} samples")
    elif SPATIAL_DATA_PATH.exists():
        print(f"Loading raw spatial data from {SPATIAL_DATA_PATH}")
        spatial_data = torch.load(SPATIAL_DATA_PATH, weights_only=False)
    else:
        print(f"Error: Spatial data not found at {SPATIAL_DATA_PATH} or {SPATIAL_DATA_PACKED_PATH}")
        return
    
    #load sequence data 
    if not TEAM_SEQUENCE_FEATURES_PATH.exists() or not TEAM_SEQUENCE_FEATURES_PATH.is_dir():
        print(f"Error: Sequence data directory not found at {TEAM_SEQUENCE_FEATURES_PATH}")
        return
    
    if args.data_limit:
        partition_files = sorted(TEAM_SEQUENCE_FEATURES_PATH.glob("part_*.parquet"))
        if not partition_files:
            raise FileNotFoundError(f"No partition files found in {TEAM_SEQUENCE_FEATURES_PATH}")
        
        target_match_ids = []
        partitions_to_load = []
        seen_match_ids = set()
        
        for part_file in partition_files:
            part_match_ids = pd.read_parquet(part_file, columns=["matchId"])["matchId"].unique()
            partition_needed = False
            for mid in part_match_ids:
                if mid not in seen_match_ids:
                    target_match_ids.append(mid)
                    seen_match_ids.add(mid)
                    partition_needed = True
                    if len(target_match_ids) >= args.data_limit:
                        target_match_ids = target_match_ids[:args.data_limit]
                        seen_match_ids = set(target_match_ids)
                        break
            if partition_needed:
                partitions_to_load.append(part_file)
            if len(target_match_ids) >= args.data_limit:
                break
        
        target_match_ids = set(target_match_ids)
        dfs = []
        for part_file in partitions_to_load:
            df_part = pd.read_parquet(part_file)
            df_part = df_part[df_part["matchId"].isin(target_match_ids)]
            if len(df_part) > 0:
                dfs.append(df_part)
        
        if not dfs:
            raise ValueError(f"No data found for the requested {args.data_limit} games")
        
        team_sequence_df = pd.concat(dfs, ignore_index=True)
        team_sequence_df = team_sequence_df.fillna(0)
        team_sequence_df = team_sequence_df[team_sequence_df["matchId"].isin(target_match_ids)]
        
        #filter spatial data if packed 
        if isinstance(spatial_data, dict) and "meta" in spatial_data:
            def normalize_match_id(mid):
                mid_str = str(mid)
                if mid_str.endswith('.0'):
                    mid_str = mid_str[:-2]
                return mid_str
            
            match_hashes = {_hash_match_id(normalize_match_id(mid)) for mid in target_match_ids}
            meta_np = spatial_data["meta"].numpy()
            valid_indices = []
            for i in range(len(meta_np)):
                if int(meta_np[i, 0]) in match_hashes:
                    valid_indices.append(i)
            
            if len(valid_indices) < len(meta_np):
                print(f"Filtering packed data to {len(valid_indices):,} samples...")
                valid_indices = np.array(valid_indices)
                spatial_data = {
                    "meta": spatial_data["meta"][valid_indices],
                    "obj_status": spatial_data["obj_status"][valid_indices],
                    "towers": spatial_data["towers"][valid_indices],
                    "kills_val": spatial_data["kills_val"],
                    "kills_idx": spatial_data["kills_idx"][valid_indices],
                    "trails_val": spatial_data["trails_val"],
                    "trails_idx": spatial_data["trails_idx"][valid_indices],
                    "num_samples": len(valid_indices),
                }
    else:
        team_sequence_df = pd.read_parquet(TEAM_SEQUENCE_FEATURES_PATH)
        team_sequence_df = team_sequence_df.fillna(0)
    
    with open(TEAM_SEQUENCE_METADATA_PATH, "r") as meta_file:
        team_sequence_meta = json.load(meta_file)
    
    team_label_cols = team_sequence_meta["team_label_cols"]
    team_eligibility_cols = team_sequence_meta.get("team_eligibility_cols", [])
    
    EXPECTED_LABEL_ORDER = [
        "dragon_taken_next_1min",
        "dragon_taken_next_2min",
        "dragon_taken_next_3min",
        "baron_taken_next_1min",
        "baron_taken_next_2min",
        "baron_taken_next_3min",
    ]
    if team_label_cols != EXPECTED_LABEL_ORDER:
        raise ValueError(f"Label order mismatch! Expected: {EXPECTED_LABEL_ORDER}, Got: {team_label_cols}")
    
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
        cnn_channels=args.cnn_channels,
        cnn_dropout=args.cnn_dropout,
        spatial_sigma=args.spatial_sigma,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        max_epochs=args.epochs,
        label_smoothing=args.label_smoothing,
        seed=args.seed,
    )
    
    result = train_single_config(
        config, train_df, val_df, test_df,
        spatial_data, team_label_cols, team_eligibility_cols, pos_weight
    )
    
    if result:
        print(f"\nTraining complete!")
        print(f"Val PR-AUC: {result['val_macro_pr_auc']:.4f}")
        print(f"Test PR-AUC: {result['test_macro_pr_auc']:.4f}")


if __name__ == "__main__":
    main()

