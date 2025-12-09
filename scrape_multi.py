#!/usr/bin/env python3
"""
Multi-region Challenger scraper for the 2025 season.

target multiple routing pairs for faster scraping - api limits are per region

GAI has helped quite a bit with this script, due to the complex logic and minor relevance for the course. We didn't simply copy-paste, but we got help.
"""

import argparse
import datetime as dt
import json
import math
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests
from dotenv import load_dotenv


DEFAULT_RATE_LIMITS = (
    (20, 1.0),     # dev key: 20 requests per second
    (100, 120.0),  # dev key: 100 requests per 120 seconds
)

DEFAULT_ROUTING = (
    "asia:kr",      # Korea
    "europe:euw1",  # EU West
    "americas:na1", # North America
)

UTC = dt.timezone.utc


@dataclass(frozen=True)
class RoutingPair:
    region: str
    platform: str

    @property
    def regional_base(self) -> str:
        return f"https://{self.region}.api.riotgames.com"

    @property
    def platform_base(self) -> str:
        return f"https://{self.platform}.api.riotgames.com"

    @property
    def label(self) -> str:
        return f"{self.region}:{self.platform}"


class ScrapeSettings:
    def __init__(
        self,
        target_queue="",
        queue_ids_allowed=(),
        map_id=0,
        min_game_duration=0,
        start_time=0,
        end_time=0,
        fetch_timelines=False,
        max_cycles_without_gain=0,
        request_timeout=10.0,
        rate_limits=DEFAULT_RATE_LIMITS,
    ):
        self.target_queue = target_queue
        self.queue_ids_allowed = queue_ids_allowed
        self.map_id = map_id
        self.min_game_duration = min_game_duration
        self.start_time = start_time
        self.end_time = end_time
        self.fetch_timelines = fetch_timelines
        self.max_cycles_without_gain = max_cycles_without_gain
        self.request_timeout = request_timeout
        self.rate_limits = rate_limits


class MatchQuota:
    """Thread-safe tracker for shared match targets + span reporting."""

    def __init__(self, target):
        self.target = target
        self._count = 0
        self._creation_times = []
        self._lock = threading.Lock()

    def needs_more(self) -> bool:
        with self._lock:
            return self._count < self.target

    def record(self, creation_ms):
        """Register a kept match. Returns (still_need, total_count, span_str)."""
        with self._lock:
            if creation_ms is not None:
                self._creation_times.append(int(creation_ms))
            if self._count < self.target:
                self._count += 1
            span = _describe_span_from_values(self._creation_times)
            still_need = self._count < self.target
            return still_need, self._count, span


class MatchSink:
    """Stores collected matches across threads and emits checkpoints."""

    def __init__(self, checkpoint_dir, checkpoint_every):
        self._matches = []
        self._timelines = {}
        self._lock = threading.Lock()
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self._checkpoint_every = checkpoint_every
        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._next_checkpoint = (
            checkpoint_every if checkpoint_every > 0 and self.checkpoint_dir else None
        )

    def add(self, match_data, timeline):
        snapshot = None
        with self._lock:
            self._matches.append(match_data)
            match_id = match_data.get("metadata", {}).get("matchId")
            if timeline is not None and match_id:
                self._timelines[match_id] = timeline
            if self._next_checkpoint is not None and len(self._matches) >= self._next_checkpoint:
                checkpoint_total = len(self._matches)
                snapshot = (
                    checkpoint_total,
                    list(self._matches),
                    dict(self._timelines),
                )
                self._next_checkpoint += self._checkpoint_every
        return snapshot

    def snapshot(self):
        with self._lock:
            return list(self._matches), dict(self._timelines)


def _describe_span_from_values(values):
    if not values:
        return None
    start = min(values)
    end = max(values)
    if start == end:
        span_minutes = 0.0
    else:
        span_minutes = (end - start) / 60_000
    start_iso = dt.datetime.fromtimestamp(start / 1000, tz=UTC).isoformat()
    end_iso = dt.datetime.fromtimestamp(end / 1000, tz=UTC).isoformat()
    return f"{start_iso} → {end_iso} (~{span_minutes:.1f} min)"


class HostRateLimiter:
    """Minimal per-host spacing."""

    def __init__(self, min_interval=0.05):
        self._min_interval = min_interval
        self._last = {}
        self._locks = {}

    def wait(self, host):
        lock = self._locks.setdefault(host, threading.Lock())
        with lock:
            now = time.monotonic()
            last = self._last.get(host, 0.0)
            delta = now - last
            if delta < self._min_interval:
                time.sleep(self._min_interval - delta)
            self._last[host] = time.monotonic()


class RiotAPIClient:
    """Simple Riot API wrapper with host-aware rate limiting."""

    def __init__(self, api_key, routing, request_timeout, rate_limits=DEFAULT_RATE_LIMITS):
        self.routing = routing
        self._session = requests.Session()
        self._session.headers.update({"X-Riot-Token": api_key})
        self._timeout = request_timeout
        self._limiter = HostRateLimiter(rate_limits)

    def _get_json(self, url, params=None, max_retries=6):
        backoff = 2.0
        host = urlparse(url).netloc
        last_error = None
        for attempt in range(max_retries):
            self._limiter.wait(host)
            try:
                response = self._session.get(url, params=params, timeout=self._timeout)
            except requests.exceptions.Timeout as exc:
                last_error = exc
                wait = min(backoff, 30.0)
                label = getattr(self.routing, "label", host)
                print(
                    f"[{label}] Request to {host} timed out (attempt {attempt + 1}/{max_retries}); "
                    f"retrying in {wait:.1f}s."
                )
                time.sleep(wait)
                backoff = min(backoff * 1.5, 30.0)
                continue
            except requests.exceptions.ConnectionError as exc:
                last_error = exc
                wait = min(backoff, 30.0)
                label = getattr(self.routing, "label", host)
                print(
                    f"[{label}] Connection error to {host} (attempt {attempt + 1}/{max_retries}); "
                    f"retrying in {wait:.1f}s."
                )
                time.sleep(wait)
                backoff = min(backoff * 1.5, 30.0)
                continue
            if response.status_code in (429, 500, 502, 503, 504):
                retry_after = response.headers.get("Retry-After")
                wait = float(retry_after) if retry_after else backoff
                time.sleep(wait)
                backoff = min(backoff * 1.5, 30.0)
                continue
            response.raise_for_status()
            return response.json()
        if last_error is not None:
            raise RuntimeError(
                f"Failed after {max_retries} attempts (last error: {last_error}) "
                f"for {url} params={params}"
            ) from last_error
        raise RuntimeError(f"Failed after {max_retries} attempts: {url} params={params}")

    def get_challenger_puuids(self, queue):
        url = f"{self.routing.platform_base}/lol/league/v4/challengerleagues/by-queue/{queue}"
        league = self._get_json(url)
        entries = league.get("entries", [])
        print(f"[{self.routing.label}] Challenger entries fetched: {len(entries)}")
        puuids = []
        failures = []
        for idx, entry in enumerate(entries, start=1):
            puuid = entry.get("puuid")
            if puuid:
                puuids.append(puuid)
            else:
                failures.append(entry)
            if idx % 50 == 0 or idx == len(entries):
                print(f"[{self.routing.label}] Processed {idx}/{len(entries)} entries (resolved {len(puuids)})")
        if failures:
            print(f"[{self.routing.label}] Failed to resolve {len(failures)} entries.")
        return puuids

    def puuid_to_match_ids(self, puuid, *, count=100, start=0, start_time=None, end_time=None, queue=None, match_type="ranked"):
        url = f"{self.routing.regional_base}/lol/match/v5/matches/by-puuid/{puuid}/ids"
        params = {"count": count, "start": start}
        if start_time is not None:
            params["startTime"] = int(start_time)
        if end_time is not None:
            params["endTime"] = int(end_time)
        if queue is not None:
            params["queue"] = int(queue)
        if match_type is not None:
            params["type"] = match_type
        return self._get_json(url, params=params)

    def get_match(self, match_id):
        url = f"{self.routing.regional_base}/lol/match/v5/matches/{match_id}"
        return self._get_json(url)

    def get_timeline(self, match_id):
        url = f"{self.routing.regional_base}/lol/match/v5/matches/{match_id}/timeline"
        return self._get_json(url)


def match_passes_filters(settings, match_data):
    info = match_data.get("info", {})
    queue_id = info.get("queueId")
    map_id = info.get("mapId")
    duration = info.get("gameDuration") or 0
    creation_ms = info.get("gameCreation")
    if creation_ms is None:
        return False
    creation_s = creation_ms // 1000
    if queue_id not in settings.queue_ids_allowed:
        return False
    if map_id != settings.map_id:
        return False
    if duration < settings.min_game_duration:
        return False
    if creation_s < settings.start_time or creation_s >= settings.end_time:
        return False
    return True


def describe_span(matches):
    creation_values = [
        m.get("info", {}).get("gameCreation")
        for m in matches
        if m.get("info", {}).get("gameCreation") is not None
    ]
    return _describe_span_from_values([int(v) for v in creation_values if v is not None])


def collect_challenger_matches(
    client: RiotAPIClient,
    settings: ScrapeSettings,
    quota: MatchQuota,
    match_sink: MatchSink,
) -> int:
    local_count = 0
    seen_ids: set[str] = set()
    routing_meta = {
        "region": client.routing.region,
        "platform": client.routing.platform,
        "label": client.routing.label,
    }

    challenger_puuids = client.get_challenger_puuids(settings.target_queue)
    offsets = {puuid: 0 for puuid in challenger_puuids}
    cycles_without_gain = 0
    cycle_number = 0

    while quota.needs_more() and cycles_without_gain < settings.max_cycles_without_gain:
        cycle_number += 1
        cycle_gain = 0
        print(
            f"[{client.routing.label}] Cycle {cycle_number} starting "
            f"(collected locally: {local_count})"
        )
        if not quota.needs_more():
            break
        for puuid in challenger_puuids:
            if not quota.needs_more():
                break
            start_index = offsets.get(puuid, 0)
            match_ids = client.puuid_to_match_ids(
                puuid,
                count=100,
                start=start_index,
                start_time=settings.start_time,
                end_time=settings.end_time,
                queue=None,
                match_type="ranked",
            )
            offsets[puuid] = start_index + len(match_ids)
            if not match_ids:
                continue
            for match_id in match_ids:
                if not quota.needs_more():
                    break
                if match_id in seen_ids:
                    continue
                match_data = client.get_match(match_id)
                if not match_passes_filters(settings, match_data):
                    continue
                match_data["_routing"] = routing_meta
                seen_ids.add(match_id)
                local_count += 1
                cycle_gain += 1
                timeline_obj = None
                if settings.fetch_timelines:
                    timeline_obj = client.get_timeline(match_id)
                    timeline_obj["_routing"] = routing_meta
                snapshot = match_sink.add(match_data, timeline_obj)
                if snapshot and match_sink.checkpoint_dir:
                    checkpoint_total, ckpt_matches, ckpt_timelines = snapshot
                    persist_checkpoint_snapshot(
                        match_sink.checkpoint_dir,
                        checkpoint_total,
                        ckpt_matches,
                        ckpt_timelines,
                    )
                info = match_data.get("info", {})
                creation_ms = info.get("gameCreation")
                still_need, global_total, global_span = quota.record(creation_ms)
                if local_count % 50 == 0:
                    creation_iso = (
                        dt.datetime.fromtimestamp(creation_ms / 1000, tz=UTC).isoformat()
                        if creation_ms
                        else "n/a"
                    )
                    span_msg = f" | span {global_span}" if global_span else ""
                    print(
                        f"[{client.routing.label}] Collected {local_count} matches "
                        f"(latest: {match_id}, duration: {info.get('gameDuration')}s, created {creation_iso})"
                        f" | total {global_total}/{quota.target}{span_msg}"
                    )
                if not still_need or not quota.needs_more():
                    break
            if not quota.needs_more():
                break
        if cycle_gain == 0:
            cycles_without_gain += 1
        else:
            cycles_without_gain = 0

    if quota.needs_more():
        print(
            f"[{client.routing.label}] Stopped after {cycle_number} cycles with {local_count} matches "
            f"(no new qualifying games)."
        )
    else:
        print(
            f"[{client.routing.label}] Reached global target contribution with {local_count} matches "
            f"in {cycle_number} cycles."
        )
    return local_count


def match_info_to_row(match_data):
    info = match_data.get("info", {})
    metadata = match_data.get("metadata", {})
    routing = match_data.get("_routing", {})
    return {
        "matchId": metadata.get("matchId"),
        "gameCreation": info.get("gameCreation"),
        "gameDuration": info.get("gameDuration"),
        "gameEndTimestamp": info.get("gameEndTimestamp"),
        "queueId": info.get("queueId"),
        "gameMode": info.get("gameMode"),
        "gameType": info.get("gameType"),
        "gameVersion": info.get("gameVersion"),
        "mapId": info.get("mapId"),
        "platformId": info.get("platformId"),
        "routingRegion": routing.get("region"),
        "routingPlatform": routing.get("platform"),
        "routingLabel": routing.get("label"),
    }


def participant_mapping_from_match(match_data):
    mapping = {}
    for participant in match_data.get("info", {}).get("participants", []):
        pid = int(participant.get("participantId"))
        mapping[pid] = {
            "championId": participant.get("championId"),
            "championName": participant.get("championName"),
            "teamId": participant.get("teamId"),
            "teamPosition": participant.get("teamPosition"),
            "win": participant.get("win"),
            "puuid": participant.get("puuid"),
            "summonerName": participant.get("summonerName"),
        }
    return mapping


def timeline_frames_to_rows(timeline, pmap):
    rows = []
    match_id = timeline.get("metadata", {}).get("matchId")
    for frame in timeline.get("info", {}).get("frames", []):
        ts = frame.get("timestamp")
        for pid, pdata in frame.get("participantFrames", {}).items():
            pos = pdata.get("position") or {}
            pid_int = int(pid)
            info = pmap.get(pid_int, {})
            rows.append(
                {
                    "matchId": match_id,
                    "frameTs": ts,
                    "participantId": pid_int,
                    "championId": info.get("championId"),
                    "championName": info.get("championName"),
                    "teamId": info.get("teamId"),
                    "x": pos.get("x"),
                    "y": pos.get("y"),
                    "level": pdata.get("level"),
                    "currentGold": pdata.get("currentGold"),
                    "totalGold": pdata.get("totalGold"),
                    "xp": pdata.get("xp"),
                    "minionsKilled": pdata.get("minionsKilled"),
                    "jungleMinionsKilled": pdata.get("jungleMinionsKilled"),
                }
            )
    return rows


OBJECTIVE_EVENT_TYPES = {
    "ELITE_MONSTER_KILL",
    "BUILDING_KILL",
    "INHIBITOR_KILL",
    "TURRET_PLATE_DESTROYED",
    "DRAGON_SOUL_GIVEN",
    "OBJECTIVE_BOUNTY_PRESTART",
    "OBJECTIVE_BOUNTY_FINISH",
}

EVENT_KEEP = {
    "CHAMPION_KILL",
    "WARD_PLACED",
    "WARD_KILL",
    *OBJECTIVE_EVENT_TYPES,
}


def timeline_events_to_rows(timeline, pmap):
    rows = []
    match_id = timeline.get("metadata", {}).get("matchId")
    for frame in timeline.get("info", {}).get("frames", []):
        ts = frame.get("timestamp")
        for event in frame.get("events", []):
            etype = event.get("type")
            if etype not in EVENT_KEEP:
                continue
            pos = event.get("position") or {}
            kid = event.get("killerId")
            vid = event.get("victimId")
            kinfo = pmap.get(int(kid)) if isinstance(kid, int) else None
            vinfo = pmap.get(int(vid)) if isinstance(vid, int) else None
            rows.append(
                {
                    "matchId": match_id,
                    "frameTs": ts,
                    "eventType": etype,
                    "timestamp": event.get("timestamp"),
                    "killerId": kid,
                    "killerChampion": (kinfo or {}).get("championName"),
                    "victimId": vid,
                    "victimChampion": (vinfo or {}).get("championName"),
                    "creatorId": event.get("creatorId"),
                    "participantId": event.get("participantId"),
                    "assistingParticipantIds": event.get("assistingParticipantIds"),
                    "x": pos.get("x"),
                    "y": pos.get("y"),
                    "monsterType": event.get("monsterType"),
                    "monsterSubType": event.get("monsterSubType"),
                    "buildingType": event.get("buildingType"),
                    "towerType": event.get("towerType"),
                    "laneType": event.get("laneType"),
                    "itemId": event.get("itemId"),
                    "wardType": event.get("wardType"),
                    "teamId": event.get("teamId"),
                }
            )
    return rows


def build_dataframes(matches, timelines):
    df_matches = pd.DataFrame([match_info_to_row(m) for m in matches])

    participants_rows = []
    for m in matches:
        match_id = m.get("metadata", {}).get("matchId")
        routing = m.get("_routing", {})
        for participant in m.get("info", {}).get("participants", []):
            participants_rows.append(
                {
                    "matchId": match_id,
                    "routingRegion": routing.get("region"),
                    "routingPlatform": routing.get("platform"),
                    "routingLabel": routing.get("label"),
                    **participant,
                }
            )
    df_participants = pd.DataFrame(participants_rows)

    df_frames = None
    df_events = None
    if timelines:
        matches_by_id = {
            m.get("metadata", {}).get("matchId"): m for m in matches
        }
        all_frames = []
        all_events = []
        for match_id, timeline in timelines.items():
            match_data = matches_by_id.get(match_id)
            if not match_data:
                continue
            pmap = participant_mapping_from_match(match_data)
            all_frames.extend(timeline_frames_to_rows(timeline, pmap))
            all_events.extend(timeline_events_to_rows(timeline, pmap))
        df_frames = pd.DataFrame(all_frames) if all_frames else None
        df_events = pd.DataFrame(all_events) if all_events else None

    return df_matches, df_participants, df_frames, df_events


def persist_dataframes(output_dir, df_matches, df_participants, df_frames, df_events):
    output_dir.mkdir(parents=True, exist_ok=True)

    matches_path = output_dir / "challenger_matches.parquet"
    df_matches.to_parquet(matches_path, index=False)
    print(f"Saved matches: {matches_path}")

    participants_path = output_dir / "challenger_participants.parquet"
    df_participants.to_parquet(participants_path, index=False)
    print(f"Saved participants: {participants_path}")

    if df_frames is not None:
        frames_path = output_dir / "challenger_frames.parquet"
        df_frames.to_parquet(frames_path, index=False)
        print(f"Saved frames: {frames_path}")

    if df_events is not None:
        events_path = output_dir / "challenger_events.parquet"
        df_events.to_parquet(events_path, index=False)
        print(f"Saved events: {events_path}")

    # CSV samples
    df_matches.head().to_csv(output_dir / "challenger_matches_sample.csv", index=False)
    df_participants.head().to_csv(output_dir / "challenger_participants_sample.csv", index=False)

    if df_frames is not None and not df_frames.empty:
        df_frames.head(500).to_csv(output_dir / "challenger_frames_sample.csv", index=False)

    if df_events is not None and not df_events.empty:
        df_events_sorted = df_events.sort_values("timestamp") if "timestamp" in df_events else df_events
        n = min(500, len(df_events_sorted))
        if n > 0:
            idx = (
                pd.Series(range(n)) * (len(df_events_sorted) - 1) / (n - 1 if n > 1 else 1)
            ).round().astype(int)
            df_events_sorted.iloc[idx.values].to_csv(
                output_dir / "challenger_events_sample.csv",
                index=False,
            )


def persist_checkpoint_snapshot(
    checkpoint_dir,
    checkpoint_total,
    matches,
    timelines,
):
    if not matches:
        return
    label = f"checkpoint_{checkpoint_total:05d}"
    target_dir = checkpoint_dir / label
    df_matches, df_participants, df_frames, df_events = build_dataframes(matches, timelines)
    persist_dataframes(target_dir, df_matches, df_participants, df_frames, df_events)
    print(f"Checkpoint saved: {target_dir} ({checkpoint_total} matches)")


def parse_routing_pairs(values):
    pairs = []
    for value in values:
        if ":" not in value:
            raise argparse.ArgumentTypeError(f"Routing '{value}' must look like region:platform (e.g. asia:kr).")
        region, platform = value.split(":", 1)
        pairs.append(RoutingPair(region=region.strip(), platform=platform.strip()))
    return pairs


def parse_timestamp(value: str) -> int:
    dt_obj = dt.datetime.fromisoformat(value)
    if dt_obj.tzinfo is None:
        dt_obj = dt_obj.replace(tzinfo=UTC)
    return int(dt_obj.timestamp())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-region Challenger ranked scraper.")
    parser.add_argument(
        "--routing",
        nargs="+",
        default=list(DEFAULT_ROUTING),
        help="REGION:PLATFORM pairs to scrape (default: %(default)s).",
    )
    parser.add_argument(
        "--target-matches",
        type=int,
        default=33000,
        help="Total matches to collect across all routing pairs (default: %(default)s).",
    )
    parser.add_argument(
        "--queue-ids",
        type=int,
        nargs="+",
        default=[420],
        help="Queue IDs to allow (default: %(default)s).",
    )
    parser.add_argument(
        "--map-id",
        type=int,
        default=11,
        help="Required map ID (default: %(default)s for Summoner's Rift).",
    )
    parser.add_argument(
        "--min-duration",
        type=int,
        default=15 * 60,
        help="Minimum game duration in seconds (default: %(default)s ≈15 min).",
    )
    parser.add_argument(
        "--start",
        type=str,
        default="2025-01-01T00:00:00+00:00",
        help="Inclusive UTC start timestamp (ISO-8601).",
    )
    parser.add_argument(
        "--end",
        type=str,
        default="2026-01-01T00:00:00+00:00",
        help="Exclusive UTC end timestamp (ISO-8601).",
    )
    parser.add_argument(
        "--target-queue",
        type=str,
        default="RANKED_SOLO_5x5",
        help="League queue to pull Challenger entries from (default: %(default)s).",
    )
    parser.add_argument(
        "--max-cycles-without-gain",
        type=int,
        default=3,
        help="Abort after this many empty cycles per routing pair (default: %(default)s).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="out_challenger",
        help="Directory to store outputs (default: %(default)s).",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Directory to write partial checkpoints (default: OUTPUT_DIR/checkpoints).",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=500,
        help="Save a checkpoint after this many collected matches (0 disables).",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=30.0,
        help="HTTP request timeout in seconds (default: %(default)s).",
    )
    parser.add_argument(
        "--fetch-timelines",
        action="store_true",
        default=False,
        help="Fetch timelines for each kept match (default: False).",
    )
    return parser.parse_args()


def summarize_results(all_matches, output_dir):
    summary = {}
    for match in all_matches:
        routing = match.get("_routing", {})
        key = routing.get("label", "unknown")
        summary.setdefault(key, 0)
        summary[key] += 1
    summary_path = output_dir / "routing_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Saved routing summary: {summary_path}")


def ensure_parquet_engine():
    return


def main() -> None:
    load_dotenv(override=True)
    ensure_parquet_engine()
    args = parse_args()
    routing_pairs = parse_routing_pairs(args.routing)

    api_key = os.getenv("RIOT_API_KEY")

    start_time = parse_timestamp(args.start)
    end_time = parse_timestamp(args.end)

    output_dir = Path(args.output_dir)
    checkpoint_every = max(0, args.checkpoint_every)
    checkpoint_dir = (
        Path(args.checkpoint_dir) if args.checkpoint_dir else output_dir / "checkpoints"
    )
    match_sink = MatchSink(
        checkpoint_dir if checkpoint_every > 0 else None,
        checkpoint_every,
    )
    settings = ScrapeSettings(
        target_queue=args.target_queue,
        queue_ids_allowed=tuple(args.queue_ids),
        map_id=args.map_id,
        min_game_duration=args.min_duration,
        start_time=start_time,
        end_time=end_time,
        fetch_timelines=args.fetch_timelines,
        max_cycles_without_gain=args.max_cycles_without_gain,
        request_timeout=args.request_timeout,
        rate_limits=DEFAULT_RATE_LIMITS,
    )

    quota = MatchQuota(args.target_matches)
    print(
        f"Routing pairs: {[pair.label for pair in routing_pairs]} "
        f"(shared target: {args.target_matches} matches)"
    )

    def scrape_pair(routing):
        client = RiotAPIClient(
            api_key=api_key,
            routing=routing,
            request_timeout=settings.request_timeout,
            rate_limits=settings.rate_limits,
        )
        count = collect_challenger_matches(client, settings, quota, match_sink)
        return routing, count

    with ThreadPoolExecutor(max_workers=len(routing_pairs)) as executor:
        futures = {executor.submit(scrape_pair, routing): routing for routing in routing_pairs}
        for future in as_completed(futures):
            routing = futures[future]
            try:
                _, count = future.result()
            except Exception as exc:  # noqa: BLE001 - surfaced to console
                print(f"[{routing.label}] scraping failed: {exc}")
                continue

    matches_snapshot, timelines_snapshot = match_sink.snapshot()
    total_matches = len(matches_snapshot)

    if total_matches < args.target_matches:
        print(
            f"Warning: only collected {total_matches} matches across all routings "
            f"(target was {args.target_matches})."
        )
    elif total_matches > args.target_matches:
        matches_snapshot.sort(
            key=lambda m: m.get("info", {}).get("gameCreation", 0),
            reverse=True,
        )
        kept = matches_snapshot[: args.target_matches]
        kept_ids = {m.get("metadata", {}).get("matchId") for m in kept}
        matches_snapshot = kept
        timelines_snapshot = {
            match_id: timeline
            for match_id, timeline in timelines_snapshot.items()
            if match_id in kept_ids
        }
        print(f"Trimmed dataset to the newest {len(matches_snapshot)} matches.")

    span_info = describe_span(matches_snapshot)
    if span_info:
        print(f"Total matches kept: {len(matches_snapshot)} | span {span_info}")
    else:
        print(f"Total matches kept: {len(matches_snapshot)}")

    df_matches, df_participants, df_frames, df_events = build_dataframes(
        matches_snapshot, timelines_snapshot
    )
    persist_dataframes(output_dir, df_matches, df_participants, df_frames, df_events)
    summarize_results(matches_snapshot, output_dir)

    print("All done. Output directory:", output_dir)


if __name__ == "__main__":
    main()

