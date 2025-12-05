import argparse
import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
import torch

#define constants
DATA_ROOT = Path(os.getenv("DATA_ROOT", "/project/project_465002423/Deep-Learning-Project-2025/combined_data"))
OUTPUT_ROOT = Path(os.getenv("DATA_ROOT", "/project/project_465002423/Deep-Learning-Project-2025/combined_data"))
CHECKPOINT_DIR = OUTPUT_ROOT / "preprocessing_checkpoints"

#checkpoint settings
CHECKPOINT_INTERVAL = 2000  
SEQUENCE_BATCH_SIZE = 10000 

#output paths
SPATIAL_DATA_PATH = OUTPUT_ROOT / "spatial_data_raw.pt"
CHAMPION_FEATURES_PATH = OUTPUT_ROOT / "champion_minute_features.parquet"
CHAMPION_SAMPLE_PATH = OUTPUT_ROOT / "champion_minute_features_sample.csv"
TEAM_FEATURES_PATH = OUTPUT_ROOT / "team_minute_features.parquet"
TEAM_SAMPLE_PATH = OUTPUT_ROOT / "team_minute_features_sample.csv"
TEAM_SEQUENCE_FEATURES_PATH = OUTPUT_ROOT / "team_sequence_features.parquet"
TEAM_SEQUENCE_SAMPLE_PATH = OUTPUT_ROOT / "team_sequence_features_sample.csv"
TEAM_SEQUENCE_METADATA_PATH = OUTPUT_ROOT / "team_sequence_metadata.json"

#shared constants
MAP_SIZE = 15000
FRAME_INTERVAL_MS = 60_000
LOOKAHEAD_MINUTES = 3
LOOKAHEAD_MS = LOOKAHEAD_MINUTES * FRAME_INTERVAL_MS
LOOKAHEAD_SECONDS = LOOKAHEAD_MINUTES * 60

#prediction horizons
LOOKAHEAD_1MIN_MS = 1 * FRAME_INTERVAL_MS
LOOKAHEAD_1MIN_SEC = 1 * 60
LOOKAHEAD_2MIN_MS = 2 * FRAME_INTERVAL_MS
LOOKAHEAD_2MIN_SEC = 2 * 60
POSITION_DECAY = 0.6

#objective timing
DRAGON_FIRST_SPAWN_SEC = 5 * 60
DRAGON_RESPAWN_SEC_NORMAL = 5 * 60
DRAGON_RESPAWN_SEC_ELDER = 6 * 60
BARON_FIRST_SPAWN_SEC = 25 * 60
BARON_RESPAWN_SEC = 6 * 60

OBJECTIVE_MONSTER_TYPES = {
    "dragon": "DRAGON",
    "baron": "BARON_NASHOR",
    "atakhan": "ATAKHAN",
    "rift_herald": "RIFTHERALD",
    "void_grubs": "HORDE",
}

NON_ELDER_DRAGON_TYPES = ["WATER_DRAGON", "FIRE_DRAGON", "EARTH_DRAGON", "AIR_DRAGON", "CHEMICAL_DRAGON", "HEXTECH_DRAGON"]
ELDER_DRAGON_TYPE = "ELDER_DRAGON"

#static tower Locations 
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

#data loading and remove duplicates
def load_and_deduplicate(limit=None):
    """
    Load all parquet files and remove duplicates by matchId.
    Returns dataframes for matches, participants, frames, events.
    """
    print("Loading parquet files")
    matches = pd.read_parquet(DATA_ROOT / "combined_matches.parquet")
    participants = pd.read_parquet(DATA_ROOT / "combined_participants.parquet")
    frames = pd.read_parquet(DATA_ROOT / "combined_frames.parquet")
    events = pd.read_parquet(DATA_ROOT / "combined_events.parquet")
    
    print(f"Loaded: matches={len(matches)}, participants={len(participants)}, "
          f"frames={len(frames)}, events={len(events)}")
    
    original_match_count = len(matches)
    matches = matches.drop_duplicates(subset=["matchId"])
    unique_match_ids = set(matches["matchId"])
    
    print(f"Matches no dupes: {original_match_count} -> {len(matches)} "
          f"({original_match_count - len(matches)} duplicates removed)")
    
    #limit the data
    if limit:
        game_ids = matches["matchId"].unique()[:limit]
        unique_match_ids = set(game_ids)
        matches = matches[matches["matchId"].isin(unique_match_ids)]
        print(f"Limited to {limit} games")
    
    #filter other tables to only contain unique matches
    participants = participants[participants["matchId"].isin(unique_match_ids)]
    frames = frames[frames["matchId"].isin(unique_match_ids)]
    events = events[events["matchId"].isin(unique_match_ids)]
    
    print(f"Filtered: participants={len(participants)}, frames={len(frames)}, events={len(events)}")
    
    #add minute column if not in
    if 'minute' not in events.columns:
        events = events.copy()
        events['minute'] = np.rint(events["frameTs"] / FRAME_INTERVAL_MS).astype(int)
    
    if 'minute' not in frames.columns:
        frames = frames.copy()
        frames['minute'] = np.rint(frames["frameTs"] / FRAME_INTERVAL_MS).astype(int)
    
    return matches, participants, frames, events, sorted(unique_match_ids)

#spatial data prep$ 
def preprocess_spatial(matches, frames, events, game_ids, resume_from=0):
    print("\nGenerating Spatial Data") 
    print('-' * 70)
    #init directories
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    spatial_checkpoint_path = CHECKPOINT_DIR / "spatial_checkpoint.pt"
    spatial_progress_path = CHECKPOINT_DIR / "spatial_progress.json"
    
    #load existing data if resuming
    data = []
    processed_game_ids = set()
    if resume_from > 0 and spatial_checkpoint_path.exists():
        print(f"Resuming from checkpoint at game {resume_from}...")
        checkpoint = torch.load(spatial_checkpoint_path)
        data = checkpoint.get('data', [])
        processed_game_ids = set(checkpoint.get('processed_game_ids', []))
        print(f"Loaded {len(data)} existing samples from {len(processed_game_ids)} games")
    
    #compute tower locations 
    tower_locs = np.array(TOWER_LOCATIONS, dtype=np.float64)
    
    print("filtering and grouping event")
    kill_events_all = events[events['eventType'] == 'CHAMPION_KILL'][['matchId', 'minute', 'frameTs', 'x', 'y', 'killerId']].copy()
    tower_events_all = events[events['eventType'] == 'BUILDING_KILL'][['matchId', 'minute', 'frameTs', 'x', 'y']].copy()
    obj_events_all = events[events['eventType'] == 'ELITE_MONSTER_KILL'][['matchId', 'minute', 'frameTs', 'monsterType', 'killerId']].copy()
    
    print("grouping frames by matchId")
    frames_grouped = {match_id: group for match_id, group in frames.groupby('matchId', sort=False)}
    events_grouped = {match_id: group for match_id, group in events.groupby('matchId', sort=False)}
    kill_events_grouped = {match_id: group for match_id, group in kill_events_all.groupby('matchId', sort=False)}
    tower_events_grouped = {match_id: group for match_id, group in tower_events_all.groupby('matchId', sort=False)}
    obj_events_grouped = {match_id: group for match_id, group in obj_events_all.groupby('matchId', sort=False)}
    print("Grouping complete")
    
    #compute tower death index for each game
    def compute_dead_tower_indices(tower_df, up_to_minute):
        """
        Find which towers are dead at given minute using euclidian calculation
        """
        if tower_df.empty:
            return []
        subset = tower_df[tower_df['minute'] <= up_to_minute]
        if subset.empty:
            return []
        
        tx = subset['x'].values.reshape(-1, 1)
        ty = subset['y'].values.reshape(-1, 1)
        
        #distance calculation to all towers
        dist_sq = (tx - tower_locs[:, 0])**2 + (ty - tower_locs[:, 1])**2
        
        dead_indices = set()
        for i in range(len(subset)):
            valid_mask = dist_sq[i] < 150**2
            if valid_mask.any():
                best_idx = np.argmin(np.where(valid_mask, dist_sq[i], np.inf))
                if dist_sq[i, best_idx] < 150**2:
                    dead_indices.add(int(best_idx))
        
        return list(dead_indices)
    
    def save_spatial_checkpoint(data, processed_ids, game_idx):
        """
        save checkpoint for spatial preprocessing.
        """
        checkpoint = {
            'data': data,
            'processed_game_ids': list(processed_ids),
            'last_game_idx': game_idx,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        }
        torch.save(checkpoint, spatial_checkpoint_path)
        with open(spatial_progress_path, 'w') as f:
            json.dump({
                'last_game_idx': game_idx,
                'total_samples': len(data),
                'total_games_processed': len(processed_ids),
                'timestamp': checkpoint['timestamp'],
            }, f, indent=2)
        print(f"  [Checkpoint saved: {len(data)} samples from {len(processed_ids)} games]")
    
    game_count = len(game_ids)
    start_time = time.time()
    
    for game_idx, game_id in enumerate(game_ids):
        #skip already processed games when resuming
        if game_id in processed_game_ids:
            continue
        
        #progress and checkpoint logic
        if game_idx % 500 == 0:
            elapsed = time.time() - start_time
            games_done = len(processed_game_ids)
            rate = games_done / elapsed if elapsed > 0 else 0
            eta = (game_count - games_done) / rate if rate > 0 else 0
            print(f"Spatial: Processing game {game_idx}/{game_count} "
                  f"({games_done} done, {rate:.1f} games/sec, ETA: {eta/60:.1f} min)")
        
        #save checkpoint 
        if game_idx > 0 and game_idx % CHECKPOINT_INTERVAL == 0 and len(processed_game_ids) > 0:
            save_spatial_checkpoint(data, processed_game_ids, game_idx)
        
        g_frames = frames_grouped.get(game_id)
        if g_frames is None or g_frames.empty:
            continue
        
        g_events = events_grouped.get(game_id, pd.DataFrame())
        g_minutes = sorted(g_frames['minute'].unique())
        
        if len(g_minutes) == 0:
            continue
        
        #get ungrouped events for this game
        kill_events = kill_events_grouped.get(game_id, pd.DataFrame())
        tower_events = tower_events_grouped.get(game_id, pd.DataFrame())
        obj_events = obj_events_grouped.get(game_id, pd.DataFrame())
        
        #index frames by minute and participantId 
        frames_by_minute = {m: grp for m, grp in g_frames.groupby('minute', sort=False)}
        
        #labels for all minutes
        elite_monster_events = g_events[g_events['eventType'] == 'ELITE_MONSTER_KILL'][['minute', 'monsterType', 'killerId']]
        
        #dead towers at each minute
        minute_to_dead_towers = {}
        for minute in g_minutes:
            minute_to_dead_towers[minute] = compute_dead_tower_indices(tower_events, minute)
        
        #initial State
        player_trails = {pid: [] for pid in range(1, 11)}
        dragon_respawn_at = 5
        baron_respawn_at = 25
        
        #sort objective events for sequential processing
        if not obj_events.empty:
            obj_events_sorted = obj_events.sort_values('minute').to_dict('records')
        else:
            obj_events_sorted = []
        obj_event_idx = 0
        
        for minute in g_minutes:
            min_frames = frames_by_minute.get(minute, pd.DataFrame())
            current_trails_state = {}
            
            #create lookup dict for current minute positions 
            if not min_frames.empty:
                pos_df = min_frames.drop_duplicates(subset='participantId', keep='first').set_index('participantId')[['x', 'y']]
                pos_lookup = pos_df.to_dict('index')
            else:
                pos_lookup = {}
            
            for pid in range(1, 11):
                new_trail = []
                for p in player_trails[pid]:
                    p[2] *= POSITION_DECAY
                    if p[2] > 0.05:
                        new_trail.append(p)
                
                if pid in pos_lookup:
                    x, y = float(pos_lookup[pid]['x']), float(pos_lookup[pid]['y'])
                    new_trail.append([x, y, 1.0])
                
                player_trails[pid] = new_trail
                current_trails_state[pid] = [list(pt) for pt in new_trail]
            
            #tower status 
            current_dead_indices = minute_to_dead_towers[minute]
            
            #objective status 
            dragon_alive = minute >= dragon_respawn_at
            baron_alive = minute >= baron_respawn_at
            
            #process objective kills at this minute
            while obj_event_idx < len(obj_events_sorted) and obj_events_sorted[obj_event_idx]['minute'] <= minute:
                row = obj_events_sorted[obj_event_idx]
                if row['minute'] == minute:
                    if row['monsterType'] == 'DRAGON':
                        dragon_alive = False
                        dragon_respawn_at = minute + 5
                    elif row['monsterType'] == 'BARON_NASHOR':
                        baron_alive = False
                        baron_respawn_at = minute + 6
                obj_event_idx += 1
            
            #kills with decay 
            kill_list = []
            if not kill_events.empty:
                start_min = max(0, minute - 10)
                relevant_kills = kill_events[(kill_events['minute'] >= start_min) & (kill_events['minute'] <= minute)]
                if not relevant_kills.empty:
                    decay_rate = 0.92
                    ages = minute - relevant_kills['minute'].values
                    weights = decay_rate ** ages
                    valid_mask = weights > 0.1
                    if valid_mask.any():
                        valid_kills = relevant_kills[valid_mask]
                        valid_weights = weights[valid_mask]
                        teams = (valid_kills['killerId'].values > 5).astype(float)
                        for i, (_, k) in enumerate(valid_kills.iterrows()):
                            kill_list.append([float(k['x']), float(k['y']), teams[i], float(valid_weights[i])])
            
            #labels
            lbl = [0, 0, 0, 0]
            if not elite_monster_events.empty:
                future_mask = (elite_monster_events['minute'] >= minute) & (elite_monster_events['minute'] <= minute + LOOKAHEAD_MINUTES)
                future_events = elite_monster_events[future_mask]
                
                if not future_events.empty:
                    for _, row in future_events.iterrows():
                        team = 1 if row['killerId'] <= 5 else 2
                        if row['monsterType'] == 'DRAGON':
                            if team == 1: lbl[0] = 1
                            else: lbl[1] = 1
                        elif row['monsterType'] == 'BARON_NASHOR':
                            if team == 1: lbl[2] = 1
                            else: lbl[3] = 1
            
            state = {
                "matchId": game_id,
                "minute": int(minute),
                "player_trails": current_trails_state,
                "dead_towers": current_dead_indices,
                "objectives": {"dragon": dragon_alive, "baron": baron_alive},
                "kills": kill_list,
                "labels": lbl
            }
            data.append(state)
        
        processed_game_ids.add(game_id)
    
    #save final checkpoint
    save_spatial_checkpoint(data, processed_game_ids, game_count)
    
    elapsed = time.time() - start_time
    print(f"Spatial data: {len(data)} samples from {len(processed_game_ids)} games in {elapsed/60:.1f} minutes")
    return data

#sequence data prep 
def preprocess_sequence(matches, participants, frames, events, game_ids):
    """
    generate sequence data for transformer model
    """
    print("\ngenerating Sequence Data$")
    print('-'*70)
    
    #build minute level champion rows
    minute_df = frames.copy().dropna(subset=["frameTs"]).reset_index(drop=True)
    minute_df["minute"] = np.rint(minute_df["frameTs"] / FRAME_INTERVAL_MS).astype(int)
    minute_df["game_time_ms"] = minute_df["minute"] * FRAME_INTERVAL_MS
    minute_df["game_time_sec"] = minute_df["game_time_ms"] / 1000.0
    minute_df["absolute_game_time_sec"] = minute_df["game_time_sec"]
    
    minute_df = minute_df.drop(columns=["championId", "championName", "teamId"], errors="ignore")
    
    participant_cols = ["matchId", "participantId", "teamId", "teamPosition", "championId", "championName"]
    participant_info = participants[participant_cols].drop_duplicates()
    participant_info["participantId"] = participant_info["participantId"].astype(int)
    minute_df["participantId"] = minute_df["participantId"].astype(int)
    minute_df = minute_df.merge(participant_info, on=["matchId", "participantId"], how="left", validate="m:1")
    
    champ_dummies = pd.get_dummies(minute_df["championId"].astype("Int64"), prefix="champ")
    minute_df = pd.concat([minute_df, champ_dummies], axis=1)
    
    print(f"Minute level rows: {len(minute_df)}, Champion one-hot columns: {len(champ_dummies.columns)}")
    
    #team lookup for event processing 
    team_lookup_df = (
        participants[["matchId", "participantId", "teamId"]]
        .dropna().drop_duplicates()
        .assign(participantId=lambda df: df["participantId"].astype(int))
    )
    
    def vectorized_infer_team(df, team_lookup_df):
        """
        team inference using merge operations 
        """
        result = df["teamId"].copy()
        
        mask = result.isna() & df["creatorId"].notna()
        if mask.any():
            merge_df = df.loc[mask, ["matchId", "creatorId"]].copy()
            merge_df["creatorId"] = merge_df["creatorId"].astype(int)
            merged = merge_df.merge(
                team_lookup_df, 
                left_on=["matchId", "creatorId"], 
                right_on=["matchId", "participantId"], 
                how="left"
            )["teamId"]
            result.loc[mask] = merged.values
        
        #if nothing use killerId
        mask = result.isna() & df["killerId"].notna()
        if mask.any():
            merge_df = df.loc[mask, ["matchId", "killerId"]].copy()
            merge_df["killerId"] = merge_df["killerId"].astype(int)
            merged = merge_df.merge(
                team_lookup_df, 
                left_on=["matchId", "killerId"], 
                right_on=["matchId", "participantId"], 
                how="left"
            )["teamId"]
            result.loc[mask] = merged.values
        
        #if nothing try participantId
        mask = result.isna() & df["participantId"].notna()
        if mask.any():
            merge_df = df.loc[mask, ["matchId", "participantId"]].copy()
            merge_df["participantId"] = merge_df["participantId"].astype(int)
            merged = merge_df.merge(
                team_lookup_df, 
                on=["matchId", "participantId"], 
                how="left"
            )["teamId"]
            result.loc[mask] = merged.values
        
        return result
    
    #ward events 
    ward_events = events[events["eventType"].isin(["WARD_PLACED", "WARD_KILL"])].copy()
    ward_events = ward_events.dropna(subset=["timestamp"]).reset_index(drop=True)
    ward_events["eventTeamId"] = vectorized_infer_team(ward_events, team_lookup_df)
    ward_events = ward_events.dropna(subset=["eventTeamId"]).copy()
    ward_events["eventTeamId"] = ward_events["eventTeamId"].astype(int)
    print(f"Ward events with team info: {len(ward_events)}")
    
    def build_event_index(df, event_type):
        cache = defaultdict(list)
        subset = df[df["eventType"] == event_type][["matchId", "eventTeamId", "timestamp"]]
        for match_id, team_id, ts in subset.itertuples(index=False):
            cache[(match_id, team_id)].append(int(ts))
        for ts_list in cache.values():
            ts_list.sort()
        return cache
    
    ward_place_index = build_event_index(ward_events, "WARD_PLACED")
    ward_kill_index = build_event_index(ward_events, "WARD_KILL")
    
    team_minutes_df = (
        minute_df[["matchId", "teamId", "minute", "game_time_ms"]]
        .drop_duplicates()
        .sort_values(["matchId", "teamId", "minute"])
        .reset_index(drop=True)
    )
    team_minutes_df["absolute_game_time_sec"] = team_minutes_df["game_time_ms"] / 1000.0
    
    def count_events_vectorized(df, event_index, time_col, window_ms=FRAME_INTERVAL_MS):
        """
        count events in window for all rows 
        """
        results = np.zeros(len(df), dtype=np.int64)
        
        #group by matchId and teamId to process in batches
        df_indexed = df.reset_index(drop=True)
        for (match_id, team_id), group in df_indexed.groupby(["matchId", "teamId"], sort=False):
            ts_list = event_index.get((match_id, team_id), [])
            if not ts_list:
                continue
            ts_arr = np.array(ts_list, dtype=np.int64)
            times = group[time_col].values.astype(np.int64)
            
            #binary search
            hi = np.searchsorted(ts_arr, times, side='right')
            lo = np.searchsorted(ts_arr, times - window_ms, side='right')
            results[group.index] = hi - lo
        
        return results
    
    team_minutes_df["wards_placed_last_minute"] = count_events_vectorized(
        team_minutes_df, ward_place_index, "game_time_ms"
    )
    team_minutes_df["wards_killed_last_minute"] = count_events_vectorized(
        team_minutes_df, ward_kill_index, "game_time_ms"
    )
    print(f"Team-minute rows: {len(team_minutes_df)}")
    
    #death cooldowns
    minute_df = minute_df.sort_values(["matchId", "participantId", "game_time_ms"]).reset_index(drop=True)
    
    death_events = (
        events[
            (events["eventType"] == "CHAMPION_KILL")
            & events["timestamp"].notna()
            & events["victimId"].notna()
        ][["matchId", "victimId", "timestamp"]]
        .rename(columns={"victimId": "participantId"})
        .assign(participantId=lambda df: df["participantId"].astype(int), timestamp=lambda df: df["timestamp"].astype(int))
    )
    
    death_index = defaultdict(list)
    for match_id, participant_id, ts in death_events.itertuples(index=False):
        death_index[(match_id, participant_id)].append(ts)
    for ts_list in death_index.values():
        ts_list.sort()
    
    def time_since_last_vectorized(df, event_index, time_col):
        """
        compute time since last event for all rows 
        """
        results = np.full(len(df), np.inf, dtype=np.float64)
        
        df_indexed = df.reset_index(drop=True)
        for (match_id, participant_id), group in df_indexed.groupby(["matchId", "participantId"], sort=False):
            ts_list = event_index.get((match_id, participant_id), [])
            if not ts_list:
                continue
            ts_arr = np.array(ts_list, dtype=np.int64)
            times = group[time_col].values.astype(np.int64)
            
            #find index where each time would be inserted 
            idx = np.searchsorted(ts_arr, times, side='right')
            valid_mask = idx > 0
            valid_idx = idx[valid_mask] - 1
            results[group.index[valid_mask]] = (times[valid_mask] - ts_arr[valid_idx]) / 1000.0
        
        return results
    
    minute_df["time_since_last_death_sec"] = time_since_last_vectorized(
        minute_df, death_index, "game_time_ms"
    )
    max_game_seconds = float(minute_df["absolute_game_time_sec"].max())
    minute_df["time_since_last_death_sec"] = (
        minute_df["time_since_last_death_sec"].replace([np.inf, -np.inf], max_game_seconds).fillna(max_game_seconds)
    )
    print("Added time_since_last_death_sec")
    
    #epic objective cooldown
    minute_df = minute_df.sort_values(["matchId", "game_time_ms", "participantId"]).reset_index(drop=True)
    
    match_minutes_df = (
        minute_df[["matchId", "minute", "game_time_ms"]]
        .drop_duplicates()
        .sort_values(["matchId", "minute"])
        .reset_index(drop=True)
    )
    match_minutes_df["absolute_game_time_sec"] = match_minutes_df["game_time_ms"] / 1000.0
    
    epic_events = events[
        (events["eventType"] == "ELITE_MONSTER_KILL")
        & events["timestamp"].notna()
        & events["monsterType"].isin(OBJECTIVE_MONSTER_TYPES.values())
    ][["matchId", "monsterType", "monsterSubType", "timestamp", "teamId", "killerId", "creatorId", "participantId"]].copy()
    epic_events["timestamp"] = epic_events["timestamp"].astype(int)
    
    epic_events_team = epic_events.copy()
    epic_events_team["eventTeamId"] = vectorized_infer_team(epic_events_team, team_lookup_df)
    epic_events_team = epic_events_team.dropna(subset=["eventTeamId"]).copy()
    epic_events_team["eventTeamId"] = epic_events_team["eventTeamId"].astype(int)
    
    def build_match_event_index(events_df, monster_type):
        cache = defaultdict(list)
        subset = events_df[events_df["monsterType"] == monster_type][["matchId", "timestamp"]]
        for match_id, ts in subset.itertuples(index=False):
            cache[match_id].append(int(ts))
        for ts_list in cache.values():
            ts_list.sort()
        return cache
    
    def build_team_event_index(events_df, monster_type):
        cache = defaultdict(list)
        subset = events_df[events_df["monsterType"] == monster_type][["matchId", "eventTeamId", "timestamp"]]
        for match_id, team_id, ts in subset.itertuples(index=False):
            cache[(match_id, team_id)].append(int(ts))
        for ts_list in cache.values():
            ts_list.sort()
        return cache
    
    monster_match_histories = {
        label: build_match_event_index(epic_events, monster_type)
        for label, monster_type in OBJECTIVE_MONSTER_TYPES.items()
    }
    monster_team_histories = {
        label: build_team_event_index(epic_events_team, monster_type)
        for label, monster_type in OBJECTIVE_MONSTER_TYPES.items()
    }
    
    team_future_dragon = monster_team_histories["dragon"]
    team_future_baron = monster_team_histories["baron"]
    
    def time_since_match_event_vectorized(df, event_index, time_col):
        """
        compute time since last event for all rows on match level.
        """
        results = np.full(len(df), np.inf, dtype=np.float64)
        
        df_indexed = df.reset_index(drop=True)
        for match_id, group in df_indexed.groupby("matchId", sort=False):
            ts_list = event_index.get(match_id, [])
            if not ts_list:
                continue
            ts_arr = np.array(ts_list, dtype=np.int64)
            times = group[time_col].values.astype(np.int64)
            
            idx = np.searchsorted(ts_arr, times, side='right')
            valid_mask = idx > 0
            valid_idx = idx[valid_mask] - 1
            results[group.index[valid_mask]] = (times[valid_mask] - ts_arr[valid_idx]) / 1000.0
        
        return results
    
    for label, history in monster_match_histories.items():
        col = f"time_since_last_{label}_sec"
        match_minutes_df[col] = time_since_match_event_vectorized(match_minutes_df, history, "game_time_ms")
    
    match_duration_cap = float(match_minutes_df["absolute_game_time_sec"].max())
    time_since_cols = [f"time_since_last_{label}_sec" for label in OBJECTIVE_MONSTER_TYPES.keys()]
    for col in time_since_cols:
        match_minutes_df[col] = (
            match_minutes_df[col].replace([np.inf, -np.inf], match_duration_cap).fillna(match_duration_cap)
        )
    
    def cumulative_kills_vectorized(df, event_index, time_col):
        """
        count cumulative events up to current time for all rows
        """
        results = np.zeros(len(df), dtype=np.int64)
        
        df_indexed = df.reset_index(drop=True)
        for (match_id, team_id), group in df_indexed.groupby(["matchId", "teamId"], sort=False):
            ts_list = event_index.get((match_id, team_id), [])
            if not ts_list:
                continue
            ts_arr = np.array(ts_list, dtype=np.int64)
            times = group[time_col].values.astype(np.int64)
            results[group.index] = np.searchsorted(ts_arr, times, side='right')
        
        return results
    
    for label, history in monster_team_histories.items():
        col = f"{label}_kills"
        team_minutes_df[col] = cumulative_kills_vectorized(team_minutes_df, history, "game_time_ms")
    
    #dragon type tracking
    dragon_events = epic_events[epic_events["monsterType"] == "DRAGON"].copy()
    dragon_events = dragon_events.sort_values(["matchId", "timestamp"]).reset_index(drop=True)
    
    dragon_match_history = defaultdict(list)
    for match_id, ts, subtype in dragon_events[["matchId", "timestamp", "monsterSubType"]].itertuples(index=False):
        dragon_match_history[match_id].append((int(ts), subtype))
    for match_id in dragon_match_history:
        dragon_match_history[match_id].sort(key=lambda x: x[0])
    
    #compute match_teams for each match 
    match_teams_map = defaultdict(set)
    for (m_id, team_id) in monster_team_histories["dragon"].keys():
        match_teams_map[m_id].add(team_id)
    
    def get_next_dragon_types_vectorized(df, dragon_history, dragon_team_history, match_teams_map):
        """
        compute next dragon type 
        """
        results = []
        df_indexed = df.reset_index(drop=True)
        
        #pre compute per match data
        match_data = {}
        for match_id in df["matchId"].unique():
            dragon_kills = dragon_history.get(match_id, [])
            dragon_ts = np.array([ts for ts, _ in dragon_kills], dtype=np.int64) if dragon_kills else np.array([], dtype=np.int64)
            dragon_types = [dtype for _, dtype in dragon_kills]
            third_type = dragon_types[2] if len(dragon_types) >= 3 else None
            
            team_ids = list(match_teams_map.get(match_id, []))
            team_ts_arrays = []
            for team_id in team_ids:
                ts_list = dragon_team_history.get((match_id, team_id), [])
                team_ts_arrays.append(np.array(ts_list, dtype=np.int64) if ts_list else np.array([], dtype=np.int64))
            
            match_data[match_id] = (dragon_ts, dragon_types, third_type, team_ts_arrays)
        
        for match_id, group in df_indexed.groupby("matchId", sort=False):
            dragon_ts, dragon_types, third_type, team_ts_arrays = match_data.get(match_id, (np.array([]), [], None, []))
            times = group["game_time_ms"].values.astype(np.int64)
            
            group_results = []
            #count kills up to now
            for t in times:
                total_kills = np.searchsorted(dragon_ts, t, side='right') if len(dragon_ts) > 0 else 0
                #max team kills
                max_team_kills = 0
                for team_ts in team_ts_arrays:
                    if len(team_ts) > 0:
                        team_kills = np.searchsorted(team_ts, t, side='right')
                        max_team_kills = max(max_team_kills, team_kills)
                
                if max_team_kills >= 4:
                    group_results.append(ELDER_DRAGON_TYPE)
                elif total_kills < 3:
                    if total_kills < len(dragon_types):
                        next_type = dragon_types[total_kills]
                        if next_type and next_type != ELDER_DRAGON_TYPE:
                            group_results.append(next_type)
                        else:
                            group_results.append(random.choice(NON_ELDER_DRAGON_TYPES))
                    else:
                        group_results.append(random.choice(NON_ELDER_DRAGON_TYPES))
                else:
                    if third_type and third_type != ELDER_DRAGON_TYPE:
                        group_results.append(third_type)
                    else:
                        group_results.append(random.choice(NON_ELDER_DRAGON_TYPES))
            
            for idx, res in zip(group.index, group_results):
                results.append((idx, res))
        results.sort(key=lambda x: x[0])
        return [r[1] for r in results]
    
    team_minutes_df["next_dragon_type"] = get_next_dragon_types_vectorized(
        team_minutes_df, dragon_match_history, monster_team_histories["dragon"], match_teams_map
    )
    
    dragon_type_dummies = pd.get_dummies(team_minutes_df["next_dragon_type"], prefix="next_dragon")
    team_minutes_df = pd.concat([team_minutes_df, dragon_type_dummies], axis=1)
    team_minutes_df = team_minutes_df.drop(columns=["next_dragon_type"])
    
    #spawn state functions 
    def _spawn_state_for_dragon_fast(timestamps, kill_times_ms, first_spawn, dragon_types, team_ts_arrays):
        """
        dragon spawn state computation
        """
        kill_times_sec = np.array(kill_times_ms, dtype=np.float64) / 1000.0 if kill_times_ms else np.array([], dtype=np.float64)
        n = len(timestamps)
        alive_vals = np.zeros(n, dtype=np.float32)
        respawn_vals = np.zeros(n, dtype=np.float32)
        next_spawn = first_spawn
        kill_idx = 0
        n_kills = len(kill_times_sec)
        
        for idx in range(n):
            t = timestamps[idx]
            current_time_ms = int(t * 1000)
            total_kills = np.searchsorted(kill_times_sec, t + 1e-6, side='right') if n_kills > 0 else 0
            max_team_kills = 0
            for team_ts in team_ts_arrays:
                if len(team_ts) > 0:
                    max_team_kills = max(max_team_kills, np.searchsorted(team_ts, current_time_ms, side='right'))
            
            if max_team_kills >= 4:
                respawn = DRAGON_RESPAWN_SEC_ELDER
            else:
                respawn = DRAGON_RESPAWN_SEC_NORMAL
            
            while kill_idx < n_kills and kill_times_sec[kill_idx] <= t + 1e-6:
                kill_time_ms = int(kill_times_sec[kill_idx] * 1000) + 1
                #check max team kills after this kill
                post_kill_max = 0
                for team_ts in team_ts_arrays:
                    if len(team_ts) > 0:
                        post_kill_max = max(post_kill_max, np.searchsorted(team_ts, kill_time_ms, side='right'))
                respawn_after = DRAGON_RESPAWN_SEC_ELDER if post_kill_max >= 4 else DRAGON_RESPAWN_SEC_NORMAL
                next_spawn = kill_times_sec[kill_idx] + respawn_after
                kill_idx += 1
            
            alive_vals[idx] = 1.0 if t >= next_spawn else 0.0
            respawn_vals[idx] = max(0.0, next_spawn - t)
        
        return alive_vals, respawn_vals
    
    def _spawn_state_for_match_fast(timestamps, kill_times_ms, first_spawn, respawn):
        """
        spawn state computation for baron/other objectives
        """
        kill_times_sec = np.array(kill_times_ms, dtype=np.float64) / 1000.0 if kill_times_ms else np.array([], dtype=np.float64)
        n = len(timestamps)
        alive_vals = np.zeros(n, dtype=np.float32)
        respawn_vals = np.zeros(n, dtype=np.float32)
        next_spawn = first_spawn
        kill_idx = 0
        n_kills = len(kill_times_sec)
        
        for idx in range(n):
            t = timestamps[idx]
            while kill_idx < n_kills and kill_times_sec[kill_idx] <= t + 1e-6:
                next_spawn = kill_times_sec[kill_idx] + respawn
                kill_idx += 1
            alive_vals[idx] = 1.0 if t >= next_spawn else 0.0
            respawn_vals[idx] = max(0.0, next_spawn - t)
        
        return alive_vals, respawn_vals
    
    #add dragon spawn columns
    alive_col = []
    respawn_col = []
    for match_id, group in match_minutes_df.groupby("matchId", sort=False):
        timestamps = group["absolute_game_time_sec"].to_numpy()
        
        #get team timestamp array for this match
        team_ids = list(match_teams_map.get(match_id, []))
        team_ts_arrays = []
        for team_id in team_ids:
            ts_list = monster_team_histories["dragon"].get((match_id, team_id), [])
            team_ts_arrays.append(np.array(ts_list, dtype=np.int64) if ts_list else np.array([], dtype=np.int64))
        
        dragon_kills = dragon_match_history.get(match_id, [])
        dragon_types = [dtype for _, dtype in dragon_kills]
        
        alive_vals, respawn_vals = _spawn_state_for_dragon_fast(
            timestamps,
            monster_match_histories["dragon"].get(match_id, []),
            DRAGON_FIRST_SPAWN_SEC,
            dragon_types,
            team_ts_arrays,
        )
        alive_col.append(pd.Series(alive_vals, index=group.index))
        respawn_col.append(pd.Series(respawn_vals, index=group.index))
    match_minutes_df["dragon_alive"] = pd.concat(alive_col).sort_index()
    match_minutes_df["dragon_respawn_in_sec"] = pd.concat(respawn_col).sort_index()
    
    #add baron spawn columns
    alive_col = []
    respawn_col = []
    for match_id, group in match_minutes_df.groupby("matchId", sort=False):
        timestamps = group["absolute_game_time_sec"].to_numpy()
        alive_vals, respawn_vals = _spawn_state_for_match_fast(
            timestamps,
            monster_match_histories["baron"].get(match_id, []),
            BARON_FIRST_SPAWN_SEC, BARON_RESPAWN_SEC,
        )
        alive_col.append(pd.Series(alive_vals, index=group.index))
        respawn_col.append(pd.Series(respawn_vals, index=group.index))
    match_minutes_df["baron_alive"] = pd.concat(alive_col).sort_index()
    match_minutes_df["baron_respawn_in_sec"] = pd.concat(respawn_col).sort_index()
    
    print("Added epic objective cooldowns")
    
    #lookahead labels
    def event_in_future_window_vectorized(df, event_index, time_col, horizon_ms=LOOKAHEAD_MS):
        """
        Check if any event occurs in (current_time, current_time + horizon] for all rows.
        """
        results = np.zeros(len(df), dtype=np.int64)
        
        df_indexed = df.reset_index(drop=True)
        for (match_id, team_id), group in df_indexed.groupby(["matchId", "teamId"], sort=False):
            ts_list = event_index.get((match_id, team_id), [])
            if not ts_list:
                continue
            ts_arr = np.array(ts_list, dtype=np.int64)
            times = group[time_col].values.astype(np.int64)
            
            hi = np.searchsorted(ts_arr, times + horizon_ms, side='right')
            lo = np.searchsorted(ts_arr, times, side='right')
            results[group.index] = (hi > lo).astype(np.int64)
        
        return results
    
    #generate labels for 1, 2, and 3 minute horizons
    team_minutes_df["dragon_taken_next_1min"] = event_in_future_window_vectorized(
        team_minutes_df, team_future_dragon, "game_time_ms", horizon_ms=LOOKAHEAD_1MIN_MS
    )
    team_minutes_df["dragon_taken_next_2min"] = event_in_future_window_vectorized(
        team_minutes_df, team_future_dragon, "game_time_ms", horizon_ms=LOOKAHEAD_2MIN_MS
    )
    team_minutes_df["dragon_taken_next_3min"] = event_in_future_window_vectorized(
        team_minutes_df, team_future_dragon, "game_time_ms", horizon_ms=LOOKAHEAD_MS
    )
    team_minutes_df["baron_taken_next_1min"] = event_in_future_window_vectorized(
        team_minutes_df, team_future_baron, "game_time_ms", horizon_ms=LOOKAHEAD_1MIN_MS
    )
    team_minutes_df["baron_taken_next_2min"] = event_in_future_window_vectorized(
        team_minutes_df, team_future_baron, "game_time_ms", horizon_ms=LOOKAHEAD_2MIN_MS
    )
    team_minutes_df["baron_taken_next_3min"] = event_in_future_window_vectorized(
        team_minutes_df, team_future_baron, "game_time_ms", horizon_ms=LOOKAHEAD_MS
    )
    
    print(f"Positive dragon labels (1min/2min/3min): "
          f"{int(team_minutes_df['dragon_taken_next_1min'].sum())}/"
          f"{int(team_minutes_df['dragon_taken_next_2min'].sum())}/"
          f"{int(team_minutes_df['dragon_taken_next_3min'].sum())}")
    print(f"Positive baron labels (1min/2min/3min): "
          f"{int(team_minutes_df['baron_taken_next_1min'].sum())}/"
          f"{int(team_minutes_df['baron_taken_next_2min'].sum())}/"
          f"{int(team_minutes_df['baron_taken_next_3min'].sum())}")
    
    #assemble feature tables
    champ_one_hot_cols = [col for col in minute_df.columns if col.startswith("champ_")]
    
    champion_stat_cols = [
        "matchId", "participantId", "teamId", "teamPosition", "minute",
        "absolute_game_time_sec", "championId", "championName", "level",
        "currentGold", "totalGold", "xp", "minionsKilled", "jungleMinionsKilled",
        "time_since_last_death_sec",
    ]
    
    missing_cols = [col for col in champion_stat_cols if col not in minute_df.columns]
    if missing_cols:
        print(f"Warning: Missing columns: {missing_cols}")
        champion_stat_cols = [col for col in champion_stat_cols if col in minute_df.columns]
    
    champion_features_df = minute_df[champion_stat_cols + champ_one_hot_cols].copy()
    champion_features_df = champion_features_df.sort_values(["matchId", "participantId", "minute"]).reset_index(drop=True)
    
    time_since_match_cols = [f"time_since_last_{label}_sec" for label in OBJECTIVE_MONSTER_TYPES.keys()]
    team_match_cols = [
        "matchId", "minute", *time_since_match_cols,
        "dragon_alive", "dragon_respawn_in_sec", "baron_alive", "baron_respawn_in_sec",
    ]
    
    team_features_df = team_minutes_df.merge(
        match_minutes_df[team_match_cols], on=["matchId", "minute"], how="left", validate="m:1",
    )
    team_features_df = team_features_df.sort_values(["matchId", "teamId", "minute"]).reset_index(drop=True)
    
    for label in OBJECTIVE_MONSTER_TYPES.keys():
        time_since_col = f"time_since_last_{label}_sec"
        kills_col = f"{label}_kills"
        if time_since_col in team_features_df.columns and kills_col in team_features_df.columns:
            #set time_since to 0 when no kills yet
            mask = team_features_df[kills_col] == 0
            team_features_df.loc[mask, time_since_col] = 0.0
    
    def _objective_eligibility(df, alive_col, respawn_col, horizon_sec):
        alive = df[alive_col]
        respawn = df[respawn_col]
        return ((alive > 0.5) | (respawn <= horizon_sec)).astype(np.float32)
    
    #generate eligibility columns for all three horizons
    team_features_df["dragon_eligible_next_1min"] = _objective_eligibility(
        team_features_df, "dragon_alive", "dragon_respawn_in_sec", LOOKAHEAD_1MIN_SEC
    )
    team_features_df["dragon_eligible_next_2min"] = _objective_eligibility(
        team_features_df, "dragon_alive", "dragon_respawn_in_sec", LOOKAHEAD_2MIN_SEC
    )
    team_features_df["dragon_eligible_next_3min"] = _objective_eligibility(
        team_features_df, "dragon_alive", "dragon_respawn_in_sec", LOOKAHEAD_SECONDS
    )
    team_features_df["baron_eligible_next_1min"] = _objective_eligibility(
        team_features_df, "baron_alive", "baron_respawn_in_sec", LOOKAHEAD_1MIN_SEC
    )
    team_features_df["baron_eligible_next_2min"] = _objective_eligibility(
        team_features_df, "baron_alive", "baron_respawn_in_sec", LOOKAHEAD_2MIN_SEC
    )
    team_features_df["baron_eligible_next_3min"] = _objective_eligibility(
        team_features_df, "baron_alive", "baron_respawn_in_sec", LOOKAHEAD_SECONDS
    )
    team_eligibility_cols = [
        "dragon_eligible_next_1min", "dragon_eligible_next_2min", "dragon_eligible_next_3min",
        "baron_eligible_next_1min", "baron_eligible_next_2min", "baron_eligible_next_3min"
    ]
    
    print(f"Champion feature shape: {champion_features_df.shape}")
    print(f"Team feature shape: {team_features_df.shape}")
    
    #build team sequence features
    role_priority = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]
    observed_positions = [pos for pos in champion_features_df["teamPosition"].dropna().unique().tolist()]
    role_order = [pos for pos in role_priority if pos in observed_positions]
    role_order += [pos for pos in observed_positions if pos not in role_order]
    if not role_order:
        role_order = role_priority.copy()
    role_prefix = {role: role.lower() for role in role_order}
    
    champ_sequence_cols = [
        "level", "currentGold", "totalGold", "xp", "minionsKilled",
        "jungleMinionsKilled", "time_since_last_death_sec", *champ_one_hot_cols,
    ]
    champ_sequence_cols = [c for c in champ_sequence_cols if c in minute_df.columns]
    
    monster_time_cols = [f"time_since_last_{label}_sec" for label in OBJECTIVE_MONSTER_TYPES.keys()]
    monster_kill_cols = [f"{label}_kills" for label in OBJECTIVE_MONSTER_TYPES.keys()]
    next_dragon_cols = [col for col in team_features_df.columns if col.startswith("next_dragon_")]
    team_context_cols = [
        "wards_placed_last_minute", "wards_killed_last_minute",
        *monster_time_cols, "dragon_alive", "dragon_respawn_in_sec",
        "baron_alive", "baron_respawn_in_sec", *monster_kill_cols, *next_dragon_cols,
    ]
    team_context_cols = [c for c in team_context_cols if c in team_features_df.columns]
    team_sequence_labels = [
        "dragon_taken_next_1min", "dragon_taken_next_2min", "dragon_taken_next_3min",
        "baron_taken_next_1min", "baron_taken_next_2min", "baron_taken_next_3min"
    ]
    
    team_merge_cols = ["matchId", "teamId", "minute", *team_context_cols, *team_sequence_labels, *team_eligibility_cols]
    team_base = (
        team_features_df[team_merge_cols]
        .sort_values(["matchId", "teamId", "minute"])
        .set_index(["matchId", "teamId", "minute"])
    )
    
    champ_subset = (
        champion_features_df[["matchId", "teamId", "minute", "teamPosition", "participantId", *champ_sequence_cols]]
        .dropna(subset=["teamPosition"]).copy()
    )
    champ_subset = (
        champ_subset.sort_values(["matchId", "teamId", "minute", "teamPosition", "participantId"])
        .drop_duplicates(subset=["matchId", "teamId", "minute", "teamPosition"], keep="first")
    )
    champ_subset = champ_subset.drop(columns=["participantId"])
    champ_subset["teamPosition"] = pd.Categorical(champ_subset["teamPosition"], categories=role_order, ordered=True)
    
    expected_columns = pd.MultiIndex.from_product([champ_sequence_cols, role_order], names=["feature", "teamPosition"])
    if champ_subset.empty:
        champ_wide = pd.DataFrame(0.0, index=team_base.index, columns=expected_columns)
    else:
        champ_wide = (
            champ_subset.set_index(["matchId", "teamId", "minute", "teamPosition"])[champ_sequence_cols]
            .unstack("teamPosition")
        )
        champ_wide = champ_wide.reindex(columns=expected_columns, fill_value=0.0)
        champ_wide = champ_wide.reindex(team_base.index, fill_value=0.0)
    
    champ_wide.columns = [f"{role_prefix[team_pos]}_{feat}" for feat, team_pos in champ_wide.columns]
    
    team_sequence_df = team_base.join(champ_wide).reset_index()
    team_sequence_df = team_sequence_df.sort_values(["matchId", "teamId", "minute"]).reset_index(drop=True)
    
    team_role_feature_cols = [f"{role_prefix[role]}_{feat}" for role in role_order for feat in champ_sequence_cols]
    team_sequence_feature_cols = team_role_feature_cols + team_context_cols
    
    print("Adding opponent features for All-Knowing perspective")
    state_features = team_role_feature_cols + team_context_cols
    
    # Exclude columns that should not be duplicated
    exclude_cols = {"matchId", "minute", "teamId"}
    exclude_cols.update(team_sequence_labels)
    exclude_cols.update(team_eligibility_cols)

    state_features = [col for col in state_features if col in team_sequence_df.columns]
    print(f"State features to duplicate: {len(state_features)} columns")
    
    #split into Team 100 and Team 200
    df_100 = team_sequence_df[team_sequence_df["teamId"] == 100].copy()
    df_200 = team_sequence_df[team_sequence_df["teamId"] == 200].copy()
    
    #create opponent feature copies for Team 100 
    opp_features_200 = df_200[["matchId", "minute"] + state_features].copy()
    opp_features_200.columns = ["matchId", "minute"] + [f"opp_{col}" for col in state_features]
    
    #create opponent feature copies for Team 200
    opp_features_100 = df_100[["matchId", "minute"] + state_features].copy()
    opp_features_100.columns = ["matchId", "minute"] + [f"opp_{col}" for col in state_features]
    
    df_100 = df_100.merge(opp_features_200, on=["matchId", "minute"], how="left", validate="m:1")
    df_200 = df_200.merge(opp_features_100, on=["matchId", "minute"], how="left", validate="m:1")
    opp_cols = [f"opp_{col}" for col in state_features]
    df_100[opp_cols] = df_100[opp_cols].fillna(0)
    df_200[opp_cols] = df_200[opp_cols].fillna(0)
    
    #concatenate back together
    team_sequence_df = pd.concat([df_100, df_200], ignore_index=True)
    team_sequence_df = team_sequence_df.sort_values(["matchId", "teamId", "minute"]).reset_index(drop=True)
    
    #update metadata to include opponent columns
    opp_feature_cols = [f"opp_{col}" for col in state_features]
    team_sequence_feature_cols = team_sequence_feature_cols + opp_feature_cols
    
    print(f"Added {len(opp_feature_cols)} opponent feature columns")
    print(f"Total feature columns: {len(team_sequence_feature_cols)}")
    
    team_sequence_meta = {
        "role_order": role_order,
        "role_prefix": role_prefix,
        "team_context_cols": team_context_cols,
        "team_label_cols": team_sequence_labels,
        "team_role_feature_cols": team_role_feature_cols,
        "team_sequence_feature_cols": team_sequence_feature_cols,
        "team_eligibility_cols": team_eligibility_cols,
    }
    
    return champion_features_df, team_features_df, team_sequence_df, team_sequence_meta

def main(limit=None, skip_spatial=False, skip_sequence=False, resume=False):
    """
    Generates both spatial and sequence data from deduplicated games
    """
    spatial_progress_path = CHECKPOINT_DIR / "spatial_progress.json"
    resume_from = 0
    
    if resume and spatial_progress_path.exists():
        with open(spatial_progress_path, 'r') as f:
            progress = json.load(f)
        resume_from = progress.get('last_game_idx', 0)
        print(f"\nResuming from checkpoint")
        print('-' * 70)
        print(f"Last checkpoint: game {resume_from}, {progress.get('total_samples', 0)} samples")
        print(f"Checkpoint time: {progress.get('timestamp', 'unknown')}")
    
    #load and deduplicate data
    matches, participants, frames, events, game_ids = load_and_deduplicate(limit)
    
    #save the list of game IDs for reference
    game_ids_path = OUTPUT_ROOT / "processed_game_ids.json"
    with open(game_ids_path, "w") as f:
        json.dump(game_ids, f)
    print(f"\nSaved {len(game_ids)} game IDs to {game_ids_path}")
    
    # generate spatial data
    if not skip_spatial:
        spatial_data = preprocess_spatial(matches, frames, events, game_ids, resume_from=resume_from)
        torch.save(spatial_data, SPATIAL_DATA_PATH)
        print(f"Saved spatial data: {SPATIAL_DATA_PATH}")
        
        #clean up checkpoint after successful completion
        if (CHECKPOINT_DIR / "spatial_checkpoint.pt").exists():
            (CHECKPOINT_DIR / "spatial_checkpoint.pt").unlink()
            print("Cleaned up spatial checkpoint")
    
    #generate sequence data
    if not skip_sequence:
        #proces in batches 
        batch_size = globals().get('SEQUENCE_BATCH_SIZE', 10000)
        total_games = len(game_ids)
        num_batches = (total_games + batch_size - 1) // batch_size
        
        print(f"\nProcessing sequence data in {num_batches} batches (batch size: {batch_size}))")
        
        all_champion_dfs = []
        all_team_dfs = []
        all_sequence_dfs = []
        sequence_meta = None
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, total_games)
            batch_game_ids = game_ids[start_idx:end_idx]
            batch_game_set = set(batch_game_ids)
            
            print(f"\n Batch {batch_idx + 1}/{num_batches}: Games {start_idx + 1}-{end_idx} ({len(batch_game_ids)} games)")
            
            #filter data to current batch
            batch_matches = matches[matches["matchId"].isin(batch_game_set)].copy()
            batch_participants = participants[participants["matchId"].isin(batch_game_set)].copy()
            batch_frames = frames[frames["matchId"].isin(batch_game_set)].copy()
            batch_events = events[events["matchId"].isin(batch_game_set)].copy()
            
            #process batch
            champion_df, team_df, sequence_df, batch_meta = preprocess_sequence(
                batch_matches, batch_participants, batch_frames, batch_events, batch_game_ids
            )
            
            #store metadata 
            sequence_meta = batch_meta
            
            #collect results
            all_champion_dfs.append(champion_df)
            all_team_dfs.append(team_df)
            all_sequence_dfs.append(sequence_df)
            
            del batch_matches, batch_participants, batch_frames, batch_events
            del champion_df, team_df, sequence_df, batch_meta
            
            print(f"Batch {batch_idx + 1} complete. Memory usage: {len(all_champion_dfs)} batches collected")
        
        #concatenate all batches
        print(f"\nConcatenating {num_batches} batches")
        
        #for champion_df, ensure all columns are aligned
        if all_champion_dfs:
            all_champion_cols = set()
            for df in all_champion_dfs:
                all_champion_cols.update(df.columns)
            all_champion_cols = sorted(all_champion_cols)
            
            aligned_champion_dfs = []
            for df in all_champion_dfs:
                aligned_df = df.reindex(columns=all_champion_cols, fill_value=0)
                #convert champion columns to uint8 
                champ_cols = [c for c in aligned_df.columns if c.startswith('champ_') or c.startswith('_champ_')]
                if champ_cols:
                    aligned_df[champ_cols] = aligned_df[champ_cols].astype('uint8')
                aligned_champion_dfs.append(aligned_df)
            champion_df = pd.concat(aligned_champion_dfs, ignore_index=True)
        else:
            champion_df = pd.DataFrame()
        
        if all_sequence_dfs:
            all_sequence_cols = set()
            for df in all_sequence_dfs:
                all_sequence_cols.update(df.columns)
            all_sequence_cols = sorted(all_sequence_cols)
            
            aligned_sequence_dfs = []
            for df in all_sequence_dfs:
                aligned_df = df.reindex(columns=all_sequence_cols, fill_value=0)
                champ_cols = [c for c in aligned_df.columns 
                             if (c.startswith('champ_') or c.startswith('_champ_') or 
                                 any(c.startswith(role + 'champ_') for role in ['top_', 'jungle_', 'middle_', 'bottom_', 'utility_']))]
                if champ_cols:
                    for col in champ_cols:
                        if col in aligned_df.columns:
                            aligned_df[col] = pd.to_numeric(aligned_df[col], errors='coerce').fillna(0).astype('uint8')
                aligned_sequence_dfs.append(aligned_df)
            sequence_df = pd.concat(aligned_sequence_dfs, ignore_index=True)
            
            champ_cols = [c for c in sequence_df.columns 
                         if (c.startswith('champ_') or c.startswith('_champ_') or 
                             any(c.startswith(role + 'champ_') for role in ['top_', 'jungle_', 'middle_', 'bottom_', 'utility_']))]
            if champ_cols:
                print(f"Converting {len(champ_cols)} champion columns to uint8...")
                for col in champ_cols:
                    if col in sequence_df.columns:
                        sequence_df[col] = pd.to_numeric(sequence_df[col], errors='coerce').fillna(0).astype('uint8')
        else:
            sequence_df = pd.DataFrame()
        
        team_df = pd.concat(all_team_dfs, ignore_index=True)
        del all_champion_dfs, all_team_dfs, all_sequence_dfs
        print(f"Final sizes: champion={len(champion_df)}, team={len(team_df)}, sequence={len(sequence_df)}")
        
        #save results
        champion_df.to_parquet(CHAMPION_FEATURES_PATH, index=False)
        print(f"Saved champion features: {CHAMPION_FEATURES_PATH}")
        champion_df.head(500).to_csv(CHAMPION_SAMPLE_PATH, index=False)
        
        team_df.to_parquet(TEAM_FEATURES_PATH, index=False)
        print(f"Saved team features: {TEAM_FEATURES_PATH}")
        team_df.head(500).to_csv(TEAM_SAMPLE_PATH, index=False)
        
        sequence_df.to_parquet(TEAM_SEQUENCE_FEATURES_PATH, index=False)
        print(f"Saved team sequence features: {TEAM_SEQUENCE_FEATURES_PATH}")
        sequence_df.head(500).to_csv(TEAM_SEQUENCE_SAMPLE_PATH, index=False)
        
        with open(TEAM_SEQUENCE_METADATA_PATH, "w") as f:
            json.dump(sequence_meta, f, indent=2)
        print(f"Saved team sequence metadata: {TEAM_SEQUENCE_METADATA_PATH}")
    
    print("\nPreprocessing Complete")
    print(f"Total unique games processed: {len(game_ids)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified preprocessing for spatial and sequence data")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of games to process")
    parser.add_argument("--skip-spatial", action="store_true", help="Skip spatial data generation")
    parser.add_argument("--skip-sequence", action="store_true", help="Skip sequence data generation")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint if available")
    parser.add_argument("--checkpoint-interval", type=int, default=2000, 
                        help="Save checkpoint every N games (default: 2000)")
    parser.add_argument("--sequence-batch-size", type=int, default=10000,
                        help="Process sequence data in batches of N games to avoid OOM (default: 10000)")
    args = parser.parse_args()
    
    globals()['CHECKPOINT_INTERVAL'] = args.checkpoint_interval
    globals()['SEQUENCE_BATCH_SIZE'] = args.sequence_batch_size
    
    main(limit=args.limit, skip_spatial=args.skip_spatial, skip_sequence=args.skip_sequence, resume=args.resume)

