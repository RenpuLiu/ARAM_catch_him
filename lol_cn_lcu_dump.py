from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from lcu_client import LCUError, connect
from match_parser import extract_games, normalize_matches, normalize_participants
from stats import build_summary, filter_rows
from storage import (
    ensure_data_dirs,
    load_json,
    merge_rows_by_key,
    matches_csv_path,
    matches_sqlite_path,
    participants_csv_path,
    read_rows_csv,
    raw_path,
    save_json,
    summary_path,
    write_matches_csv,
    write_matches_sqlite,
    write_rows_csv,
)


def dump_lcu_data(
    n: int = 50,
    out: str | Path = "data",
    lockfile: str | None = None,
    include_timelines: bool = False,
) -> dict[str, Any]:
    root = ensure_data_dirs(out)
    _archive_existing_raw_batches(root)
    client = connect(lockfile)

    current = client.current_summoner()
    save_json(raw_path(root, "current_summoner.json"), current)

    matchlist = client.matchlist(beg_index=0, end_index=n)
    save_json(raw_path(root, "matchlist_raw.json"), matchlist)

    games = extract_games(matchlist)[:n]
    save_json(raw_path(root, "games_from_matchlist.json"), games)

    detailed: list[dict[str, Any]] = []
    timelines: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    for game in games:
        game_id = game.get("gameId") or game.get("id") or game.get("matchId")
        if not game_id:
            continue

        try:
            detail = client.game_detail(game_id)
            if isinstance(detail, dict):
                detailed.append(detail)
        except Exception as exc:  # LCU schema and availability differ between clients.
            errors.append({"game_id": str(game_id), "kind": "detail", "error": str(exc)})

        if include_timelines:
            try:
                timeline = client.game_timeline(game_id)
                if isinstance(timeline, dict):
                    timelines.append(timeline)
            except Exception as exc:
                errors.append(
                    {"game_id": str(game_id), "kind": "timeline", "error": str(exc)}
                )

    save_json(raw_path(root, "game_details_raw.json"), detailed)
    _archive_game_payloads(root, detailed, "game_details")
    if include_timelines:
        save_json(raw_path(root, "timelines_raw.json"), timelines)
        _archive_game_payloads(root, timelines, "timelines")
    if errors:
        save_json(raw_path(root, "fetch_errors.json"), errors)

    source_games = detailed if detailed else games
    rows = normalize_matches(source_games, current)
    participant_rows = normalize_participants(source_games, current)
    existing_rows = read_rows_csv(matches_csv_path(root))
    existing_participant_rows = read_rows_csv(participants_csv_path(root))
    existing_match_ids = _match_ids(existing_rows)
    fetched_match_ids = _match_ids(rows)
    merged_rows = merge_rows_by_key(existing_rows, rows, ["match_id"])
    merged_participant_rows = merge_rows_by_key(
        existing_participant_rows,
        participant_rows,
        ["match_id", "participant_id"],
    )

    added_match_ids = fetched_match_ids - existing_match_ids
    updated_match_ids = fetched_match_ids & existing_match_ids

    write_matches_csv(matches_csv_path(root), merged_rows)
    write_rows_csv(participants_csv_path(root), merged_participant_rows)
    write_matches_sqlite(matches_sqlite_path(root), merged_rows, merged_participant_rows)

    summary = {
        "all_queues": build_summary(merged_rows),
        "aram": build_summary(filter_rows(merged_rows, aram_only=True)),
        "raw_games": len(games),
        "detailed_games": len(detailed),
        "fetched_matches": len(rows),
        "fetched_participants": len(participant_rows),
        "added_matches": len(added_match_ids),
        "updated_matches": len(updated_match_ids),
        "total_matches": len(merged_rows),
        "total_participants": len(merged_participant_rows),
        "normalized_matches": len(merged_rows),
        "normalized_participants": len(merged_participant_rows),
        "errors": errors,
    }
    save_json(summary_path(root), summary)

    return {
        "data_dir": str(root),
        "lockfile": str(client.lockfile.path),
        "current_summoner": current,
        "raw_games": len(games),
        "detailed_games": len(detailed),
        "fetched_matches": len(rows),
        "fetched_participants": len(participant_rows),
        "added_matches": len(added_match_ids),
        "updated_matches": len(updated_match_ids),
        "total_matches": len(merged_rows),
        "total_participants": len(merged_participant_rows),
        "normalized_matches": len(merged_rows),
        "normalized_participants": len(merged_participant_rows),
        "summary": summary,
        "errors": errors,
    }


def _archive_existing_raw_batches(root: Path) -> None:
    detail_batch = load_json(raw_path(root, "game_details_raw.json"), default=[])
    if isinstance(detail_batch, list):
        _archive_game_payloads(root, detail_batch, "game_details")

    timeline_batch = load_json(raw_path(root, "timelines_raw.json"), default=[])
    if isinstance(timeline_batch, list):
        _archive_game_payloads(root, timeline_batch, "timelines")


def _archive_game_payloads(root: Path, games: list[dict[str, Any]], dirname: str) -> None:
    if not games:
        return
    for game in games:
        if not isinstance(game, dict):
            continue
        game_id = _game_id(game)
        if not game_id:
            continue
        save_json(raw_path(root, f"{dirname}/{game_id}.json"), game)


def _game_id(game: dict[str, Any]) -> str:
    value = game.get("gameId") or game.get("id") or game.get("matchId")
    return str(value) if value is not None else ""


def _match_ids(rows: list[dict[str, Any]]) -> set[str]:
    return {str(row.get("match_id")).strip() for row in rows if str(row.get("match_id", "")).strip()}


def main() -> int:
    parser = argparse.ArgumentParser(description="Dump local League Client match history.")
    parser.add_argument("--n", type=int, default=50, help="number of recent matches")
    parser.add_argument("--out", type=str, default="data", help="output directory")
    parser.add_argument("--lockfile", type=str, default=None, help="manual lockfile path")
    parser.add_argument(
        "--timeline",
        action="store_true",
        help="also fetch timelines; slower but useful for later analysis",
    )
    args = parser.parse_args()

    try:
        result = dump_lcu_data(
            n=args.n,
            out=args.out,
            lockfile=args.lockfile,
            include_timelines=args.timeline,
        )
    except LCUError as exc:
        print(f"LCU error: {exc}")
        return 2

    current = result.get("current_summoner") or {}
    print("Current summoner:")
    print(json.dumps(current, ensure_ascii=False, indent=2)[:1200])
    print()
    print(f"Lockfile: {result['lockfile']}")
    print(f"Raw games: {result['raw_games']}")
    print(f"Detailed games: {result['detailed_games']}")
    print(f"Fetched normalized matches: {result['fetched_matches']}")
    print(f"Fetched participants: {result['fetched_participants']}")
    print(f"Added matches: {result['added_matches']}")
    print(f"Updated/deduped matches: {result['updated_matches']}")
    print(f"Historical matches: {result['total_matches']}")
    print(f"Historical participants: {result['total_participants']}")
    print(f"Saved to: {result['data_dir']}")
    if result["errors"]:
        print(f"Partial fetch errors: {len(result['errors'])} (see raw/fetch_errors.json)")
    print("Summary:")
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
