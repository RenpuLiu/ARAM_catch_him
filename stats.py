from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime
from statistics import mean
from typing import Any

from match_parser import is_aram_game


def filter_rows(rows: list[dict[str, Any]], aram_only: bool = True) -> list[dict[str, Any]]:
    if not aram_only:
        return rows
    return [row for row in rows if _row_is_aram(row)]


def sort_recent(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: _to_int(row.get("game_creation_ms")) or 0, reverse=True)


def build_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    wins = sum(1 for row in rows if _to_bool(row.get("win")))
    losses = total - wins
    recent = sort_recent(rows)[:20]
    recent_wins = sum(1 for row in recent if _to_bool(row.get("win")))

    return {
        "games": total,
        "wins": wins,
        "losses": losses,
        "winrate": _ratio(wins, total),
        "recent_20_games": len(recent),
        "recent_20_winrate": _ratio(recent_wins, len(recent)),
        "avg_kda": _avg(rows, "kda"),
        "avg_kills": _avg(rows, "kills"),
        "avg_deaths": _avg(rows, "deaths"),
        "avg_assists": _avg(rows, "assists"),
        "avg_damage_to_champions": _avg(rows, "damage_to_champions"),
        "avg_damage_taken": _avg(rows, "damage_taken"),
    }


def champion_stats(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = str(row.get("champion_name") or row.get("champion_id") or "Unknown")
        groups[key].append(row)

    result = []
    for champion, group in groups.items():
        wins = sum(1 for row in group if _to_bool(row.get("win")))
        result.append(
            {
                "champion": champion,
                "champion_id": group[0].get("champion_id", ""),
                "games": len(group),
                "wins": wins,
                "winrate": _ratio(wins, len(group)),
                "avg_kda": _avg(group, "kda"),
                "avg_damage_to_champions": _avg(group, "damage_to_champions"),
                "avg_damage_taken": _avg(group, "damage_taken"),
                "avg_deaths": _avg(group, "deaths"),
            }
        )

    return sorted(result, key=lambda row: (row["games"], row["winrate"]), reverse=True)


def hour_stats(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        hour = _hour(row.get("game_creation"))
        if hour is not None:
            groups[hour].append(row)

    result = []
    for hour in range(24):
        group = groups.get(hour, [])
        wins = sum(1 for row in group if _to_bool(row.get("win")))
        result.append(
            {
                "hour": hour,
                "games": len(group),
                "wins": wins,
                "winrate": _ratio(wins, len(group)),
            }
        )
    return result


def item_stats(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counter: Counter[int] = Counter()
    win_counter: Counter[int] = Counter()

    for row in rows:
        items = _items(row.get("items_json"))
        for item_id in items:
            counter[item_id] += 1
            if _to_bool(row.get("win")):
                win_counter[item_id] += 1

    return [
        {
            "item_id": item_id,
            "games": games,
            "wins": win_counter[item_id],
            "winrate": _ratio(win_counter[item_id], games),
        }
        for item_id, games in counter.most_common()
    ]


def format_percent(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value * 100:.1f}%"


def _avg(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [_to_float(row.get(key)) for row in rows]
    clean = [value for value in values if value is not None]
    return round(mean(clean), 2) if clean else None


def _ratio(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return round(numerator / denominator, 4)


def _hour(value: Any) -> int | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value)).hour
    except ValueError:
        return None


def _items(value: Any) -> list[int]:
    if not value:
        return []
    if isinstance(value, list):
        return [_to_int(item) for item in value if _to_int(item)]
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    return [_to_int(item) for item in parsed if _to_int(item)]


def _row_is_aram(row: dict[str, Any]) -> bool:
    return _to_bool(row.get("is_aram")) or is_aram_game(row)


def _to_bool(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, float) and value != value:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "win", "won", "victory"}
    return False


def _to_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _to_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
