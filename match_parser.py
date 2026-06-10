from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any

ARAM_QUEUE_IDS = {450, 2400}
ARAM_MAP_IDS = {12}
ARAM_GAME_MODES = {"ARAM", "KIWI"}
ITEM_KEYS = ("item0", "item1", "item2", "item3", "item4", "item5", "item6")
PARTICIPANT_EXTRA_STATS = {
    "champLevel": "champ_level",
    "goldSpent": "gold_spent",
    "totalDamageDealt": "total_damage_dealt",
    "physicalDamageDealtToChampions": "physical_damage_to_champions",
    "magicDamageDealtToChampions": "magic_damage_to_champions",
    "trueDamageDealtToChampions": "true_damage_to_champions",
    "physicalDamageTaken": "physical_damage_taken",
    "magicalDamageTaken": "magic_damage_taken",
    "trueDamageTaken": "true_damage_taken",
    "damageSelfMitigated": "damage_self_mitigated",
    "damageDealtToObjectives": "damage_to_objectives",
    "damageDealtToTurrets": "damage_to_turrets",
    "totalHeal": "total_heal",
    "totalUnitsHealed": "total_units_healed",
    "timeCCingOthers": "time_ccing_others",
    "totalTimeCrowdControlDealt": "total_time_cc_dealt",
    "visionScore": "vision_score",
    "wardsPlaced": "wards_placed",
    "wardsKilled": "wards_killed",
    "doubleKills": "double_kills",
    "tripleKills": "triple_kills",
    "quadraKills": "quadra_kills",
    "pentaKills": "penta_kills",
    "largestMultiKill": "largest_multi_kill",
    "largestKillingSpree": "largest_killing_spree",
    "largestCriticalStrike": "largest_critical_strike",
    "longestTimeSpentLiving": "longest_life_seconds",
    "turretKills": "turret_kills",
    "inhibitorKills": "inhibitor_kills",
    "neutralMinionsKilled": "neutral_minions_killed",
}
PERK_KEYS = (
    "perk0",
    "perk1",
    "perk2",
    "perk3",
    "perk4",
    "perk5",
    "perkPrimaryStyle",
    "perkSubStyle",
)
AUGMENT_KEYS = (
    "playerAugment1",
    "playerAugment2",
    "playerAugment3",
    "playerAugment4",
    "playerAugment5",
    "playerAugment6",
)


def extract_games(matchlist: Any) -> list[dict[str, Any]]:
    if isinstance(matchlist, list):
        return [game for game in matchlist if isinstance(game, dict)]

    if not isinstance(matchlist, dict):
        return []

    candidates = [
        matchlist.get("games", {}).get("games")
        if isinstance(matchlist.get("games"), dict)
        else None,
        matchlist.get("games") if isinstance(matchlist.get("games"), list) else None,
        matchlist.get("matches") if isinstance(matchlist.get("matches"), list) else None,
        matchlist.get("gameList", {}).get("games")
        if isinstance(matchlist.get("gameList"), dict)
        else None,
    ]

    for candidate in candidates:
        if isinstance(candidate, list):
            return [game for game in candidate if isinstance(game, dict)]

    return []


def normalize_matches(
    games: list[dict[str, Any]],
    current_summoner: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    rows = []
    for game in games:
        row = normalize_match(game, current_summoner)
        if row:
            rows.append(row)
    return rows


def normalize_participants(
    games: list[dict[str, Any]],
    current_summoner: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    rows = []
    for game in games:
        rows.extend(normalize_match_participants(game, current_summoner))
    return rows


def normalize_match_participants(
    game: dict[str, Any],
    current_summoner: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    own_participant = find_own_participant(game, current_summoner)
    own_id = _to_int(_first(own_participant or {}, "participantId"))
    own_team_id = _to_int(_first(own_participant or {}, "teamId"))
    participants = [
        _as_dict(participant)
        for participant in game.get("participants") or []
        if isinstance(participant, dict)
    ]

    rows = []
    for participant in participants:
        row = _participant_row(game, participant)
        participant_id = _to_int(row.get("participant_id"))
        team_id = _to_int(row.get("team_id"))
        row["is_self"] = participant_id == own_id
        if own_team_id is None:
            row["side"] = "unknown"
        elif participant_id == own_id:
            row["side"] = "self"
        elif team_id == own_team_id:
            row["side"] = "ally"
        else:
            row["side"] = "enemy"
        rows.append(row)

    return rows


def is_aram_game(game_or_row: dict[str, Any]) -> bool:
    queue_id = _to_int(_first(game_or_row, "queueId", "queue_id", "queue"))
    map_id = _to_int(_first(game_or_row, "mapId", "map_id"))
    game_mode = str(_first(game_or_row, "gameMode", "game_mode", default="")).upper()
    return (
        queue_id in ARAM_QUEUE_IDS
        or map_id in ARAM_MAP_IDS
        or game_mode in ARAM_GAME_MODES
    )


def _participant_row(game: dict[str, Any], participant: dict[str, Any]) -> dict[str, Any]:
    stats = _as_dict(participant.get("stats"))
    identity = _identity_for_participant(game, participant)
    player = _as_dict(identity.get("player")) if identity else {}
    game_id = _first(game, "gameId", "id", "matchId")
    queue_id = _to_int(_first(game, "queueId", "queue_id", "queue"))
    map_id = _to_int(_first(game, "mapId", "map_id"))
    duration_seconds = _to_int(
        _first(game, "gameDuration", "duration", "gameLength", "timePlayed")
    )
    creation_ms = _timestamp_ms(_first(game, "gameCreation", "gameCreated", "createdAt"))
    champion_id = _to_int(_first(participant, "championId", "champion_id"))
    kills = _to_int(_first(stats, "kills", default=0)) or 0
    deaths = _to_int(_first(stats, "deaths", default=0)) or 0
    assists = _to_int(_first(stats, "assists", default=0)) or 0
    item_ids = [_to_int(_first(stats, key, f"{key}Id", default=0)) or 0 for key in ITEM_KEYS]
    item_ids = [item for item in item_ids if item > 0]
    perk_ids = [_to_int(_first(stats, key, default=0)) or 0 for key in PERK_KEYS]
    perk_ids = [perk for perk in perk_ids if perk > 0]
    augment_ids = [_to_int(_first(stats, key, default=0)) or 0 for key in AUGMENT_KEYS]
    augment_ids = [augment for augment in augment_ids if augment > 0]

    row = {
        "match_id": str(game_id) if game_id is not None else "",
        "queue_id": queue_id,
        "map_id": map_id,
        "is_aram": is_aram_game(game),
        "game_mode": _first(game, "gameMode", "game_mode", default=""),
        "game_type": _first(game, "gameType", "game_type", default=""),
        "platform_id": _first(game, "platformId", "platform_id", default=""),
        "game_creation_ms": creation_ms,
        "game_creation": _format_timestamp(creation_ms),
        "duration_seconds": duration_seconds,
        "duration_minutes": round((duration_seconds or 0) / 60, 1)
        if duration_seconds is not None
        else None,
        "participant_id": _to_int(_first(participant, "participantId")),
        "team_id": _to_int(_first(participant, "teamId")),
        "champion_id": champion_id,
        "champion_name": _first_non_empty(
            participant,
            "championName",
            "champion_name",
            default=str(champion_id) if champion_id is not None else "",
        ),
        "summoner_name": _summoner_name(player),
        "riot_id": _riot_id(player),
        "tag_line": _first_non_empty(player, "tagLine", default=""),
        "account_id": _first_non_empty(player, "currentAccountId", "accountId", default=""),
        "summoner_id": _first_non_empty(player, "summonerId", "currentSummonerId", default=""),
        "puuid": _first_non_empty(player, "puuid", "currentPuuid", default=""),
        "win": _is_win(_first(stats, "win", "winner", default=False)),
        "kills": kills,
        "deaths": deaths,
        "assists": assists,
        "kda": round((kills + assists) / max(1, deaths), 2),
        "damage_to_champions": _to_int(
            _first(
                stats,
                "totalDamageDealtToChampions",
                "damageDealtToChampions",
                "magicDamageDealtToChampions",
                default=0,
            )
        )
        or 0,
        "damage_taken": _to_int(_first(stats, "totalDamageTaken", "damageTaken", default=0))
        or 0,
        "gold_earned": _to_int(_first(stats, "goldEarned", "gold", default=0)) or 0,
        "cs": (
            (_to_int(_first(stats, "totalMinionsKilled", default=0)) or 0)
            + (_to_int(_first(stats, "neutralMinionsKilled", default=0)) or 0)
        ),
        "spell1_id": _to_int(_first(participant, "spell1Id", "summoner1Id")),
        "spell2_id": _to_int(_first(participant, "spell2Id", "summoner2Id")),
        "items_json": json.dumps(item_ids, ensure_ascii=False),
        "perks_json": json.dumps(perk_ids, ensure_ascii=False),
        "augments_json": json.dumps(augment_ids, ensure_ascii=False),
        "raw_stats_json": json.dumps(stats, ensure_ascii=False, separators=(",", ":")),
    }

    for index, item in enumerate(item_ids[:7]):
        row[f"item{index}"] = item
    for key, output_key in PARTICIPANT_EXTRA_STATS.items():
        row[output_key] = _to_int(_first(stats, key, default=0)) or 0
    for key in PERK_KEYS:
        row[_snake_case(key)] = _to_int(_first(stats, key, default=0)) or 0
    for key in AUGMENT_KEYS:
        row[_snake_case(key)] = _to_int(_first(stats, key, default=0)) or 0

    return row


def _related_players(
    game: dict[str, Any], own_participant: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    own_id = _to_int(_first(own_participant, "participantId"))
    own_team_id = _to_int(_first(own_participant, "teamId"))
    teammates = []
    opponents = []

    for participant in game.get("participants") or []:
        if not isinstance(participant, dict):
            continue
        participant_id = _to_int(_first(participant, "participantId"))
        if participant_id == own_id:
            continue

        summary = _player_summary(game, participant)
        if own_team_id is not None and _to_int(_first(participant, "teamId")) == own_team_id:
            teammates.append(summary)
        else:
            opponents.append(summary)

    return teammates, opponents


def _player_summary(game: dict[str, Any], participant: dict[str, Any]) -> dict[str, Any]:
    stats = _as_dict(participant.get("stats"))
    identity = _identity_for_participant(game, participant)
    player = _as_dict(identity.get("player")) if identity else {}
    kills = _to_int(_first(stats, "kills", default=0)) or 0
    deaths = _to_int(_first(stats, "deaths", default=0)) or 0
    assists = _to_int(_first(stats, "assists", default=0)) or 0
    champion_id = _to_int(_first(participant, "championId", "champion_id"))

    return {
        "participant_id": _to_int(_first(participant, "participantId")),
        "team_id": _to_int(_first(participant, "teamId")),
        "summoner_name": _summoner_name(player),
        "riot_id": _riot_id(player),
        "champion_id": champion_id,
        "champion_name": _first_non_empty(
            participant,
            "championName",
            "champion_name",
            default=str(champion_id) if champion_id is not None else "",
        ),
        "win": _is_win(_first(stats, "win", "winner", default=False)),
        "kills": kills,
        "deaths": deaths,
        "assists": assists,
        "kda": round((kills + assists) / max(1, deaths), 2),
        "damage_to_champions": _to_int(
            _first(stats, "totalDamageDealtToChampions", "damageDealtToChampions", default=0)
        )
        or 0,
    }


def _display_name(player: dict[str, Any]) -> str:
    return (
        str(player.get("riot_id") or "").strip()
        or str(player.get("summoner_name") or "").strip()
        or str(player.get("champion_name") or player.get("champion_id") or "").strip()
    )


def normalize_match(
    game: dict[str, Any],
    current_summoner: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    participant = find_own_participant(game, current_summoner)
    if not participant:
        return None

    stats = _as_dict(participant.get("stats"))
    identity = _identity_for_participant(game, participant)
    player = _as_dict(identity.get("player")) if identity else {}

    game_id = _first(game, "gameId", "id", "matchId")
    queue_id = _to_int(_first(game, "queueId", "queue_id", "queue"))
    map_id = _to_int(_first(game, "mapId", "map_id"))
    duration_seconds = _to_int(
        _first(game, "gameDuration", "duration", "gameLength", "timePlayed")
    )
    creation_ms = _timestamp_ms(_first(game, "gameCreation", "gameCreated", "createdAt"))
    champion_id = _to_int(_first(participant, "championId", "champion_id"))
    champion_name = _first(
        participant,
        "championName",
        "champion_name",
        default=str(champion_id) if champion_id is not None else "",
    )

    kills = _to_int(_first(stats, "kills", default=0)) or 0
    deaths = _to_int(_first(stats, "deaths", default=0)) or 0
    assists = _to_int(_first(stats, "assists", default=0)) or 0
    kda = round((kills + assists) / max(1, deaths), 2)

    item_ids = [_to_int(_first(stats, key, f"{key}Id", default=0)) or 0 for key in ITEM_KEYS]
    item_ids = [item for item in item_ids if item > 0]
    teammates, opponents = _related_players(game, participant)

    row = {
        "match_id": str(game_id) if game_id is not None else "",
        "queue_id": queue_id,
        "map_id": map_id,
        "is_aram": is_aram_game(game),
        "game_mode": _first(game, "gameMode", "game_mode", default=""),
        "game_type": _first(game, "gameType", "game_type", default=""),
        "platform_id": _first(game, "platformId", "platform_id", default=""),
        "game_creation_ms": creation_ms,
        "game_creation": _format_timestamp(creation_ms),
        "duration_seconds": duration_seconds,
        "duration_minutes": round((duration_seconds or 0) / 60, 1)
        if duration_seconds is not None
        else None,
        "participant_id": _to_int(_first(participant, "participantId")),
        "team_id": _to_int(_first(participant, "teamId")),
        "champion_id": champion_id,
        "champion_name": champion_name,
        "summoner_name": _summoner_name(player, current_summoner),
        "account_id": _first(player, "currentAccountId", "accountId", default=""),
        "summoner_id": _first(player, "summonerId", "currentSummonerId", default=""),
        "puuid": _first(player, "puuid", "currentPuuid", default=""),
        "win": _is_win(_first(stats, "win", "winner", default=False)),
        "kills": kills,
        "deaths": deaths,
        "assists": assists,
        "kda": kda,
        "damage_to_champions": _to_int(
            _first(
                stats,
                "totalDamageDealtToChampions",
                "damageDealtToChampions",
                "magicDamageDealtToChampions",
                default=0,
            )
        )
        or 0,
        "damage_taken": _to_int(_first(stats, "totalDamageTaken", "damageTaken", default=0))
        or 0,
        "gold_earned": _to_int(_first(stats, "goldEarned", "gold", default=0)) or 0,
        "cs": (
            (_to_int(_first(stats, "totalMinionsKilled", default=0)) or 0)
            + (_to_int(_first(stats, "neutralMinionsKilled", default=0)) or 0)
        ),
        "spell1_id": _to_int(_first(participant, "spell1Id", "summoner1Id")),
        "spell2_id": _to_int(_first(participant, "spell2Id", "summoner2Id")),
        "items_json": json.dumps(item_ids, ensure_ascii=False),
        "teammate_names": ", ".join(_display_name(player) for player in teammates),
        "opponent_names": ", ".join(_display_name(player) for player in opponents),
        "teammates_json": json.dumps(teammates, ensure_ascii=False),
        "opponents_json": json.dumps(opponents, ensure_ascii=False),
    }

    for index, item in enumerate(item_ids[:7]):
        row[f"item{index}"] = item

    return row


def find_own_participant(
    game: dict[str, Any],
    current_summoner: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    participants = [_as_dict(p) for p in game.get("participants") or [] if isinstance(p, dict)]
    if not participants:
        if "stats" in game or "championId" in game:
            return game
        return None

    participant_id = _own_participant_id(game, current_summoner)
    if participant_id is not None:
        for participant in participants:
            if _to_int(participant.get("participantId")) == participant_id:
                return participant

    account_ids = _identity_values(current_summoner, "accountId", "currentAccountId")
    summoner_ids = _identity_values(current_summoner, "summonerId", "currentSummonerId")
    puuids = _identity_values(current_summoner, "puuid", "currentPuuid")

    for participant in participants:
        player = _as_dict(participant.get("player"))
        if _matches_any(player, account_ids, "accountId", "currentAccountId"):
            return participant
        if _matches_any(player, summoner_ids, "summonerId", "currentSummonerId"):
            return participant
        if _matches_any(player, puuids, "puuid", "currentPuuid"):
            return participant

    stat_participants = [
        p for p in participants if isinstance(p.get("stats"), dict) and "championId" in p
    ]
    if len(stat_participants) == 1:
        return stat_participants[0]

    return stat_participants[0] if stat_participants else None


def _own_participant_id(
    game: dict[str, Any],
    current_summoner: dict[str, Any] | None,
) -> int | None:
    identities = [
        _as_dict(identity)
        for identity in game.get("participantIdentities") or []
        if isinstance(identity, dict)
    ]
    if not identities or not current_summoner:
        return None

    account_ids = _identity_values(current_summoner, "accountId", "currentAccountId")
    summoner_ids = _identity_values(current_summoner, "summonerId", "currentSummonerId")
    puuids = _identity_values(current_summoner, "puuid", "currentPuuid")
    names = _identity_values(
        current_summoner,
        "displayName",
        "summonerName",
        "gameName",
        normalize=True,
    )

    for identity in identities:
        player = _as_dict(identity.get("player"))
        if _matches_any(player, account_ids, "accountId", "currentAccountId"):
            return _to_int(identity.get("participantId"))
        if _matches_any(player, summoner_ids, "summonerId", "currentSummonerId"):
            return _to_int(identity.get("participantId"))
        if _matches_any(player, puuids, "puuid", "currentPuuid"):
            return _to_int(identity.get("participantId"))
        if _matches_any(
            player,
            names,
            "summonerName",
            "gameName",
            "riotIdGameName",
            normalize=True,
        ):
            return _to_int(identity.get("participantId"))

    return None


def _identity_for_participant(
    game: dict[str, Any], participant: dict[str, Any]
) -> dict[str, Any] | None:
    participant_id = _to_int(participant.get("participantId"))
    for identity in game.get("participantIdentities") or []:
        if not isinstance(identity, dict):
            continue
        if _to_int(identity.get("participantId")) == participant_id:
            return identity
    return None


def _identity_values(
    obj: dict[str, Any] | None,
    *keys: str,
    normalize: bool = False,
) -> set[str]:
    if not obj:
        return set()
    values = set()
    for key in keys:
        value = obj.get(key)
        if value is None or value == "":
            continue
        text = str(value)
        values.add(text.strip().lower() if normalize else text)
    return values


def _matches_any(
    obj: dict[str, Any],
    values: set[str],
    *keys: str,
    normalize: bool = False,
) -> bool:
    if not values:
        return False
    for key in keys:
        value = obj.get(key)
        if value is None or value == "":
            continue
        text = str(value)
        if normalize:
            text = text.strip().lower()
        if text in values:
            return True
    return False


def _first(obj: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        value = obj.get(key)
        if value is not None:
            return value
    return default


def _first_non_empty(obj: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        value = obj.get(key)
        if value is not None and value != "":
            return value
    return default


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _to_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _timestamp_ms(value: Any) -> int | None:
    number = _to_int(value)
    if number is None:
        return None
    if number < 10_000_000_000:
        return number * 1000
    return number


def _format_timestamp(value_ms: int | None) -> str:
    if value_ms is None:
        return ""
    dt = datetime.fromtimestamp(value_ms / 1000, tz=timezone.utc).astimezone()
    return dt.isoformat(timespec="seconds")


def _is_win(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"win", "won", "true", "1", "victory"}
    return False


def _summoner_name(
    player: dict[str, Any], current_summoner: dict[str, Any] | None = None
) -> str:
    return (
        _first_non_empty(player, "summonerName", "gameName", "riotIdGameName", default="")
        or _first_non_empty(current_summoner or {}, "displayName", "summonerName", default="")
        or ""
    )


def _riot_id(player: dict[str, Any]) -> str:
    game_name = _first_non_empty(player, "gameName", "riotIdGameName", "summonerName", default="")
    tag_line = _first_non_empty(player, "tagLine", "riotIdTagLine", default="")
    if game_name and tag_line:
        return f"{game_name}#{tag_line}"
    return str(game_name or "")


def _snake_case(value: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", value).lower()
