"""
Packs the spatial data into dense tensors. Ensuring less RAM usage.
"""
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm


def hash_match_id(match_id):
    return np.int32(hash(match_id) & 0x7FFFFFFF)


def pack_spatial_data(
    spatial_data,
    output_path,
    dtype_float=torch.float16,
    dtype_int=torch.int32,
):
    N = len(spatial_data)
    meta = torch.zeros((N, 2), dtype=dtype_int)
    obj_status = torch.zeros((N, 2), dtype=dtype_float)
    towers = torch.zeros((N, 30), dtype=torch.uint8)
    kills_idx = torch.zeros((N, 2), dtype=dtype_int)
    trails_idx = torch.zeros((N, 10, 2), dtype=dtype_int)

    all_kills = []
    all_trails = []
    labels = []
    kills_offset = 0
    trails_offset = 0

    for i, state in enumerate(tqdm(spatial_data, desc="Packing")):
        meta[i, 0] = hash_match_id(state["matchId"])
        meta[i, 1] = int(state["minute"])

        objectives = state.get("objectives", {})
        obj_status[i, 0] = float(objectives.get("dragon", 0.0))
        obj_status[i, 1] = float(objectives.get("baron", 0.0))

        dead_towers = state.get("dead_towers", [])
        if dead_towers:
            towers[i, dead_towers] = 1

        kills_data = state.get("kills", [])
        if kills_data:
            kills_array = np.asarray(kills_data, dtype=np.float32).reshape(-1, 4)
            all_kills.append(kills_array)
            kills_idx[i, 0] = kills_offset
            kills_idx[i, 1] = kills_array.shape[0]
            kills_offset += kills_array.shape[0]

        trails = state.get("player_trails", {})
        for pid_str, points in trails.items():
            player_idx = int(pid_str) - 1
            points_array = np.asarray(points, dtype=np.float32).reshape(-1, 3)
            all_trails.append(points_array)
            trails_idx[i, player_idx, 0] = trails_offset
            trails_idx[i, player_idx, 1] = points_array.shape[0]
            trails_offset += points_array.shape[0]

        if "labels" in state:
            labels.append(state["labels"])

    kills_val = (
        torch.from_numpy(np.vstack(all_kills)).to(dtype_float)
        if all_kills
        else torch.empty((0, 4), dtype=dtype_float)
    )
    trails_val = (
        torch.from_numpy(np.vstack(all_trails)).to(dtype_float)
        if all_trails
        else torch.empty((0, 3), dtype=dtype_float)
    )

    labels_tensor = torch.tensor(labels, dtype=dtype_float) if labels else None

    spatial_lookup = {}
    meta_np = meta.numpy()
    for i in range(N):
        spatial_lookup[(int(meta_np[i, 0]), int(meta_np[i, 1]))] = i

    packed_data = {
        "meta": meta,
        "obj_status": obj_status,
        "towers": towers,
        "kills_val": kills_val,
        "kills_idx": kills_idx,
        "trails_val": trails_val,
        "trails_idx": trails_idx,
        "spatial_lookup": spatial_lookup,
        "num_samples": N,
    }
    if labels_tensor is not None:
        packed_data["labels"] = labels_tensor

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(packed_data, output_path)
    print(f"Saved {N} samples to {output_path}")


def main():
    input_path = Path("combined_data/spatial_data_raw.pt")
    output_path = Path("combined_data/spatial_data_packed.pt")
    dtype_float = torch.float16
    dtype_int = torch.int32

    spatial_data = torch.load(input_path)
    pack_spatial_data(spatial_data, output_path, dtype_float=dtype_float, dtype_int=dtype_int)


if __name__ == "__main__":
    main()

