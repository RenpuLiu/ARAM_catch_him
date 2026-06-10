from __future__ import annotations

import json
import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

import requests

from storage import matches_csv_path, participants_csv_path


DEFAULT_SKILL_PATH = Path("skills/aram-match-analyst/SKILL.md")
USER_SKILLS_DIR = Path("skills/user")
REPORTS_DIRNAME = "reports"
ENV_FILES = (".env", ".env.local")
DEFAULT_MAX_OUTPUT_TOKENS = 8000
DEFAULT_REASONING_EFFORT = ""


class LLMAnalysisError(RuntimeError):
    pass


@dataclass(frozen=True)
class SkillInfo:
    name: str
    description: str
    path: Path


def list_analysis_skills(root: str | Path = ".") -> list[SkillInfo]:
    base = Path(root)
    candidates = [base / DEFAULT_SKILL_PATH]
    user_dir = base / USER_SKILLS_DIR
    if user_dir.exists():
        candidates.extend(sorted(user_dir.glob("*/SKILL.md")))

    skills = []
    for path in candidates:
        if path.exists():
            metadata, _ = _read_skill(path)
            skills.append(
                SkillInfo(
                    name=metadata.get("name") or path.parent.name,
                    description=metadata.get("description") or "",
                    path=path,
                )
            )
    return skills


def load_skill_prompt(path: str | Path = DEFAULT_SKILL_PATH) -> str:
    _, body = _read_skill(Path(path))
    return body.strip()


def load_combined_skill_prompt(
    skill_paths: str | Path | list[str | Path] | tuple[str | Path, ...] | None = None,
) -> str:
    paths = _normalize_skill_paths(skill_paths)
    if len(paths) == 1:
        return load_skill_prompt(paths[0])

    sections = [
        "The following SKILL.md documents are combined. Follow all of them. "
        "When they conflict, preserve data-grounded analysis and prefer the later skill for style or formatting only."
    ]
    for index, path in enumerate(paths, start=1):
        metadata, body = _read_skill(path)
        name = metadata.get("name") or path.parent.name
        description = metadata.get("description") or ""
        sections.append(
            f"\n\n--- Skill {index}: {name} ---\n"
            f"Source: {path}\n"
            f"Description: {description}\n\n"
            f"{body.strip()}"
        )
    return "\n".join(sections).strip()


def _normalize_skill_paths(
    skill_paths: str | Path | list[str | Path] | tuple[str | Path, ...] | None,
) -> list[Path]:
    if skill_paths is None:
        return [DEFAULT_SKILL_PATH]
    if isinstance(skill_paths, (str, Path)):
        paths = [Path(skill_paths)]
    else:
        paths = [Path(path) for path in skill_paths]
    return paths or [DEFAULT_SKILL_PATH]


def load_local_env(root: str | Path = ".", override: bool = False) -> list[str]:
    loaded: list[str] = []
    base = Path(root)
    for filename in ENV_FILES:
        path = base / filename
        if not path.exists():
            continue
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = _strip_env_value(value.strip())
            if not key:
                continue
            if override or key not in os.environ:
                os.environ[key] = value
                loaded.append(key)
    return loaded


def build_analysis_payload(
    data_dir: str | Path = "data",
    min_partner_games: int = 2,
    recent_games: int = 50,
) -> dict[str, Any]:
    import pandas as pd

    data_dir = Path(data_dir)
    matches_path = matches_csv_path(data_dir)
    participants_path = participants_csv_path(data_dir)
    if not matches_path.exists():
        raise LLMAnalysisError(f"Missing {matches_path}. Refresh LCU data first.")
    if not participants_path.exists():
        raise LLMAnalysisError(f"Missing {participants_path}. Refresh LCU data first.")

    matches = pd.read_csv(matches_path)
    participants = pd.read_csv(participants_path)
    if matches.empty or participants.empty:
        raise LLMAnalysisError("No match or participant rows available.")

    matches = _sort_recent(matches).head(recent_games)
    match_ids = set(matches["match_id"].astype(str))
    participants = participants[participants["match_id"].astype(str).isin(match_ids)]
    self_rows = participants[participants.get("side", "") == "self"].copy()
    ally_rows = participants[participants.get("side", "").isin(["ally"])].copy()
    enemy_rows = participants[participants.get("side", "") == "enemy"].copy()

    team_context = _team_context(participants)
    self_rows = _attach_team_context(self_rows, team_context)
    ally_rows = _attach_team_context(ally_rows, team_context)

    partner_summaries = _partner_summaries(
        matches=matches,
        self_rows=self_rows,
        ally_rows=ally_rows,
        team_context=team_context,
        min_partner_games=min_partner_games,
    )
    player_profiles = _players_for_equal_analysis(
        self_rows=self_rows,
        ally_rows=ally_rows,
        min_partner_games=min_partner_games,
    )

    payload = {
        "metadata": {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "data_dir": str(data_dir),
            "match_count": int(len(matches)),
            "participant_count": int(len(participants)),
            "min_partner_games": int(min_partner_games),
            "recent_games_limit": int(recent_games),
            "note": "Frequent allies are inferred from repeated ally rows, not confirmed premade party data.",
        },
        "user": {
            "identity": _identity_summary(self_rows),
            "overall": _performance_summary(self_rows),
            "champions": _top_champions(self_rows),
            "spells": _top_pairs(self_rows, ["spell1_id", "spell2_id"], limit=8),
            "items": _top_items(self_rows, limit=16),
            "style_signals": _style_signals(self_rows),
            "recent_matches": _recent_match_summaries(matches, self_rows),
        },
        "players_for_equal_analysis": player_profiles,
        "recent_performance_ranking_seed": _recent_performance_ranking_seed(player_profiles),
        "frequent_allies": partner_summaries,
        "opponent_context": {
            "enemy_avg": _performance_summary(enemy_rows),
            "enemy_top_champions": _top_champions(enemy_rows, limit=12),
        },
    }
    return _json_safe(payload)


def build_llm_user_input(payload: dict[str, Any]) -> str:
    safe_payload = _json_safe(payload)
    return (
        "请基于下面的 JSON 数据做 ARAM 对局分析。重点分析用户本人、常见队友/疑似多排、"
        "搭配表现、平均表现、游戏风格和可执行建议。"
        "请对 players_for_equal_analysis 里的每个人使用相同维度分析，"
        "并基于 recent_performance_ranking_seed 与证据生成最近表现排序。\n\n"
        "数据：\n"
        f"{json.dumps(safe_payload, ensure_ascii=False, indent=2)}"
    )


def call_llm(
    system_prompt: str,
    user_input: str,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    api_style: str | None = None,
    max_output_tokens: int | None = None,
    reasoning_effort: str | None = None,
    timeout: int = 600,
) -> tuple[str, dict[str, Any]]:
    load_local_env()
    api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")
    if not api_key:
        raise LLMAnalysisError("Set OPENAI_API_KEY or LLM_API_KEY before running LLM analysis.")

    model = model or os.getenv("LLM_MODEL") or "gpt-5.4"
    base_url = (base_url or os.getenv("LLM_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
    api_style = (api_style or os.getenv("LLM_API_STYLE") or "responses").strip().lower()
    max_output_tokens = _resolve_max_output_tokens(max_output_tokens)
    reasoning_effort = _resolve_reasoning_effort(reasoning_effort)

    if api_style == "chat":
        response = _call_chat_completions(
            base_url=base_url,
            api_key=api_key,
            model=model,
            system_prompt=system_prompt,
            user_input=user_input,
            max_output_tokens=max_output_tokens,
            timeout=timeout,
        )
        text = _extract_chat_text(response)
        return _with_truncation_notice(text, response), response

    response = _call_responses(
        base_url=base_url,
        api_key=api_key,
        model=model,
        system_prompt=system_prompt,
        user_input=user_input,
        max_output_tokens=max_output_tokens,
        reasoning_effort=reasoning_effort,
        timeout=timeout,
    )
    text = _extract_response_text(response)
    return _with_truncation_notice(text, response), response


def generate_analysis_report(
    data_dir: str | Path = "data",
    skill_path: str | Path | list[str | Path] | tuple[str | Path, ...] = DEFAULT_SKILL_PATH,
    min_partner_games: int = 2,
    recent_games: int = 50,
    dry_run: bool = False,
    model: str | None = None,
    max_output_tokens: int | None = None,
    reasoning_effort: str | None = None,
) -> dict[str, Any]:
    payload = build_analysis_payload(
        data_dir=data_dir,
        min_partner_games=min_partner_games,
        recent_games=recent_games,
    )
    skill_paths = _normalize_skill_paths(skill_path)
    system_prompt = load_combined_skill_prompt(skill_paths)
    user_input = build_llm_user_input(payload)

    result: dict[str, Any] = {
        "payload": payload,
        "system_prompt_path": str(skill_paths[0]),
        "system_prompt_paths": [str(path) for path in skill_paths],
        "user_input": user_input,
        "report": "",
        "raw_response": {},
    }
    if dry_run:
        return result

    report, raw_response = call_llm(
        system_prompt=system_prompt,
        user_input=user_input,
        model=model,
        max_output_tokens=max_output_tokens,
        reasoning_effort=reasoning_effort,
    )
    result["report"] = report
    result["raw_response"] = raw_response
    save_report(data_dir, report, payload, raw_response)
    return result


def save_report(
    data_dir: str | Path,
    report: str,
    payload: dict[str, Any],
    raw_response: dict[str, Any] | None = None,
) -> Path:
    reports_dir = Path(data_dir) / REPORTS_DIRNAME
    reports_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = reports_dir / f"llm_analysis_{stamp}.md"
    json_path = reports_dir / f"llm_analysis_{stamp}.json"
    report_path.write_text(report, encoding="utf-8")
    json_path.write_text(
        json.dumps(
            _json_safe({"payload": payload, "raw_response": raw_response or {}}),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return report_path


def _call_responses(
    base_url: str,
    api_key: str,
    model: str,
    system_prompt: str,
    user_input: str,
    max_output_tokens: int,
    reasoning_effort: str,
    timeout: int,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model,
        "instructions": system_prompt,
        "input": user_input,
        "max_output_tokens": max_output_tokens,
        "store": False,
    }
    if reasoning_effort:
        body["reasoning"] = {"effort": reasoning_effort}

    response = requests.post(
        f"{base_url}/responses",
        headers=_headers(api_key),
        json=body,
        timeout=timeout,
    )
    return _checked_json(response)


def _call_chat_completions(
    base_url: str,
    api_key: str,
    model: str,
    system_prompt: str,
    user_input: str,
    max_output_tokens: int,
    timeout: int,
) -> dict[str, Any]:
    response = requests.post(
        f"{base_url}/chat/completions",
        headers=_headers(api_key),
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input},
            ],
            "max_tokens": max_output_tokens,
            "temperature": 0.2,
        },
        timeout=timeout,
    )
    return _checked_json(response)


def _checked_json(response: requests.Response) -> dict[str, Any]:
    if response.status_code >= 400:
        raise LLMAnalysisError(f"LLM API error {response.status_code}: {response.text[:1200]}")
    return response.json()


def _headers(api_key: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def _resolve_max_output_tokens(value: int | None) -> int:
    if value is not None:
        return _coerce_positive_int(value, "max_output_tokens")
    raw_value = os.getenv("LLM_MAX_OUTPUT_TOKENS")
    if raw_value is None or not raw_value.strip():
        return DEFAULT_MAX_OUTPUT_TOKENS
    return _coerce_positive_int(raw_value, "LLM_MAX_OUTPUT_TOKENS")


def _coerce_positive_int(value: Any, name: str) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise LLMAnalysisError(f"{name} must be an integer.") from exc
    if number <= 0:
        raise LLMAnalysisError(f"{name} must be greater than 0.")
    return number


def _resolve_reasoning_effort(value: str | None) -> str:
    raw_value = os.getenv("LLM_REASONING_EFFORT", DEFAULT_REASONING_EFFORT) if value is None else value
    effort = (raw_value or "").strip().lower()
    if effort in {"", "default", "none", "off", "false", "0"}:
        return ""
    return effort


def _with_truncation_notice(text: str, response: dict[str, Any]) -> str:
    notice = _truncation_notice(response)
    if not notice:
        return text
    return (
        f"> 注意：这份报告可能被截断。{notice}\n"
        "> 建议把 LLM_MAX_OUTPUT_TOKENS 调高到 8000-12000，"
        "或减少“分析局数”后重新生成。\n\n"
        f"{text}"
    )


def _truncation_notice(response: dict[str, Any]) -> str:
    status = response.get("status")
    incomplete_details = response.get("incomplete_details") or {}
    reason = incomplete_details.get("reason")
    if status == "incomplete" or reason:
        detail = f"原因是 {reason}" if reason else "状态为 incomplete"
        return f"Responses API 返回不完整输出，{detail}。"

    finish_reason = _first_chat_finish_reason(response)
    if finish_reason == "length":
        return "Chat Completions 返回 finish_reason=length，说明输出达到了长度上限。"
    return ""


def _first_chat_finish_reason(response: dict[str, Any]) -> str:
    choices = response.get("choices") or []
    if not choices:
        return ""
    return str(choices[0].get("finish_reason") or "")


def _extract_response_text(response: dict[str, Any]) -> str:
    chunks = []
    for item in response.get("output", []):
        if item.get("type") != "message":
            continue
        for content in item.get("content", []):
            if content.get("type") == "output_text":
                chunks.append(content.get("text", ""))
    text = "\n".join(chunk for chunk in chunks if chunk).strip()
    if not text:
        raise LLMAnalysisError("LLM response did not contain output_text.")
    return text


def _extract_chat_text(response: dict[str, Any]) -> str:
    choices = response.get("choices") or []
    if not choices:
        raise LLMAnalysisError("Chat completion response did not contain choices.")
    return (choices[0].get("message", {}).get("content") or "").strip()


def _read_skill(path: Path) -> tuple[dict[str, str], str]:
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---"):
        return {}, text
    _, frontmatter, body = text.split("---", 2)
    metadata: dict[str, str] = {}
    for line in frontmatter.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        metadata[key.strip()] = value.strip().strip('"').strip("'")
    return metadata, body


def _strip_env_value(value: str) -> str:
    if (
        len(value) >= 2
        and value[0] == value[-1]
        and value[0] in {'"', "'"}
    ):
        return value[1:-1]
    if " #" in value:
        value = value.split(" #", 1)[0].rstrip()
    return value


def _sort_recent(df: Any) -> Any:
    if "game_creation_ms" in df.columns:
        return df.sort_values("game_creation_ms", ascending=False)
    return df


def _team_context(participants: Any) -> dict[tuple[str, int], dict[str, float]]:
    context = {}
    for (match_id, team_id), group in participants.groupby(["match_id", "team_id"]):
        context[(str(match_id), int(team_id))] = {
            "team_kills": _sum(group, "kills"),
            "team_damage": _sum(group, "damage_to_champions"),
            "team_damage_taken": _sum(group, "damage_taken"),
            "team_gold": _sum(group, "gold_earned"),
            "team_deaths": _sum(group, "deaths"),
        }
    return context


def _attach_team_context(rows: Any, context: dict[tuple[str, int], dict[str, float]]) -> Any:
    if rows.empty:
        return rows
    enriched = rows.copy()
    damage_shares = []
    taken_shares = []
    kill_participations = []
    gold_shares = []
    for _, row in enriched.iterrows():
        key = (str(row.get("match_id")), int(_num(row.get("team_id"))))
        team = context.get(key, {})
        damage_shares.append(_ratio(_num(row.get("damage_to_champions")), team.get("team_damage")))
        taken_shares.append(_ratio(_num(row.get("damage_taken")), team.get("team_damage_taken")))
        kill_participations.append(
            _ratio(_num(row.get("kills")) + _num(row.get("assists")), team.get("team_kills"))
        )
        gold_shares.append(_ratio(_num(row.get("gold_earned")), team.get("team_gold")))
    enriched["damage_share"] = damage_shares
    enriched["damage_taken_share"] = taken_shares
    enriched["kill_participation"] = kill_participations
    enriched["gold_share"] = gold_shares
    return enriched


def _partner_summaries(
    matches: Any,
    self_rows: Any,
    ally_rows: Any,
    team_context: dict[tuple[str, int], dict[str, float]],
    min_partner_games: int,
) -> list[dict[str, Any]]:
    if ally_rows.empty:
        return []

    ally_rows = ally_rows.copy()
    ally_rows["partner_key"] = ally_rows.apply(_partner_key, axis=1)
    summaries = []
    for partner_key, partner_rows in ally_rows.groupby("partner_key"):
        if len(partner_rows) < min_partner_games:
            continue
        match_ids = set(partner_rows["match_id"].astype(str))
        user_with_partner = self_rows[self_rows["match_id"].astype(str).isin(match_ids)]
        partner_rows = _attach_team_context(partner_rows, team_context)
        together_matches = matches[matches["match_id"].astype(str).isin(match_ids)]
        summaries.append(
            {
                "partner": _partner_identity(partner_rows),
                "games_together": int(len(match_ids)),
                "wins_together": int(_bool_sum(together_matches, "win")),
                "winrate_together": _ratio(_bool_sum(together_matches, "win"), len(match_ids)),
                "user_when_together": _performance_summary(user_with_partner),
                "partner_average": _performance_summary(partner_rows),
                "combined": _combined_summary(user_with_partner, partner_rows, team_context),
                "partner_champions": _top_champions(partner_rows, limit=8),
                "user_champions_with_partner": _top_champions(user_with_partner, limit=8),
                "recent_match_ids": list(sorted(match_ids, reverse=True))[:8],
            }
        )

    return sorted(
        summaries,
        key=lambda item: (item["games_together"], item["winrate_together"] or 0),
        reverse=True,
    )[:12]


def _players_for_equal_analysis(
    self_rows: Any,
    ally_rows: Any,
    min_partner_games: int,
) -> list[dict[str, Any]]:
    profiles = [_player_profile("self", self_rows)]
    if ally_rows.empty:
        return profiles

    ally_rows = ally_rows.copy()
    ally_rows["partner_key"] = ally_rows.apply(_partner_key, axis=1)
    partner_profiles = []
    for _, partner_rows in ally_rows.groupby("partner_key"):
        if len(partner_rows) < min_partner_games:
            continue
        partner_profiles.append(_player_profile("frequent_ally", partner_rows))

    partner_profiles.sort(
        key=lambda profile: (
            profile.get("overall", {}).get("games") or 0,
            profile.get("overall", {}).get("winrate") or 0,
        ),
        reverse=True,
    )
    return profiles + partner_profiles[:12]


def _player_profile(role: str, rows: Any) -> dict[str, Any]:
    identity = _identity_summary(rows) if role == "self" else _partner_identity(rows)
    return {
        "role": role,
        "identity": identity,
        "overall": _performance_summary(rows),
        "champions": _top_champions(rows, limit=10),
        "spells": _top_pairs(rows, ["spell1_id", "spell2_id"], limit=6),
        "items": _top_items(rows, limit=12),
        "style_signals": _style_signals(rows),
        "recent_matches": _recent_player_match_summaries(rows, limit=10),
    }


def _recent_performance_ranking_seed(profiles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranking = []
    for profile in profiles:
        overall = profile.get("overall", {})
        identity = profile.get("identity", {})
        score = _recent_performance_score(overall)
        ranking.append(
            {
                "player": identity.get("riot_id") or identity.get("summoner_name") or "",
                "role": profile.get("role", ""),
                "games": overall.get("games"),
                "score": score,
                "score_note": "Heuristic seed only; final report should rank with evidence and confidence.",
                "winrate": overall.get("winrate"),
                "avg_kda": overall.get("avg_kda"),
                "avg_damage_share": overall.get("avg_damage_share"),
                "avg_damage_taken_share": overall.get("avg_damage_taken_share"),
                "avg_kill_participation": overall.get("avg_kill_participation"),
                "deaths_per_10_min": overall.get("deaths_per_10_min"),
            }
        )
    return sorted(ranking, key=lambda item: item["score"] or 0, reverse=True)


def _recent_performance_score(overall: dict[str, Any]) -> float | None:
    if not overall or not overall.get("games"):
        return None
    score = 0.0
    score += (_num(overall.get("winrate")) or 0) * 35
    score += min(_num(overall.get("avg_kda")), 8) / 8 * 15
    score += min(_num(overall.get("avg_damage_share")), 0.35) / 0.35 * 20
    score += min(_num(overall.get("avg_kill_participation")), 0.9) / 0.9 * 15
    score += min(_num(overall.get("avg_damage_taken_share")), 0.35) / 0.35 * 8
    score += min(_num(overall.get("avg_cc_time")), 40) / 40 * 5
    score -= min(_num(overall.get("deaths_per_10_min")), 8) / 8 * 8
    games = _num(overall.get("games"))
    confidence = min(games, 8) / 8
    return round(score * (0.75 + 0.25 * confidence), 2)


def _identity_summary(rows: Any) -> dict[str, Any]:
    if rows.empty:
        return {}
    row = rows.iloc[0]
    return {
        "riot_id": _clean(row.get("riot_id")),
        "summoner_name": _clean(row.get("summoner_name")),
        "summoner_id": _clean(row.get("summoner_id")),
        "puuid_present": bool(_clean(row.get("puuid"))),
    }


def _partner_identity(rows: Any) -> dict[str, Any]:
    row = rows.iloc[0]
    return {
        "riot_id": _clean(row.get("riot_id")),
        "summoner_name": _clean(row.get("summoner_name")),
        "summoner_id": _clean(row.get("summoner_id")),
        "puuid_present": bool(_clean(row.get("puuid"))),
    }


def _performance_summary(rows: Any) -> dict[str, Any]:
    if rows.empty:
        return {}
    games = len(rows)
    wins = _bool_sum(rows, "win")
    duration = _mean(rows, "duration_minutes")
    return {
        "games": int(games),
        "wins": int(wins),
        "winrate": _ratio(wins, games),
        "avg_kills": _mean(rows, "kills"),
        "avg_deaths": _mean(rows, "deaths"),
        "avg_assists": _mean(rows, "assists"),
        "avg_kda": _mean(rows, "kda"),
        "avg_duration_minutes": duration,
        "avg_damage": _mean(rows, "damage_to_champions"),
        "avg_damage_taken": _mean(rows, "damage_taken"),
        "avg_damage_mitigated": _mean(rows, "damage_self_mitigated"),
        "avg_gold": _mean(rows, "gold_earned"),
        "avg_cs": _mean(rows, "cs"),
        "avg_champ_level": _mean(rows, "champ_level"),
        "avg_heal": _mean(rows, "total_heal"),
        "avg_cc_time": _mean(rows, "time_ccing_others"),
        "avg_vision": _mean(rows, "vision_score"),
        "avg_damage_share": _mean(rows, "damage_share"),
        "avg_damage_taken_share": _mean(rows, "damage_taken_share"),
        "avg_kill_participation": _mean(rows, "kill_participation"),
        "deaths_per_10_min": _per_10(rows, "deaths"),
        "damage_per_min": _per_min(rows, "damage_to_champions"),
        "damage_taken_per_min": _per_min(rows, "damage_taken"),
    }


def _combined_summary(user_rows: Any, partner_rows: Any, team_context: dict[tuple[str, int], dict[str, float]]) -> dict[str, Any]:
    by_match = {}
    for _, row in user_rows.iterrows():
        by_match.setdefault(str(row.get("match_id")), {})["user"] = row
    for _, row in partner_rows.iterrows():
        by_match.setdefault(str(row.get("match_id")), {})["partner"] = row

    combined_damage_shares = []
    combined_taken_shares = []
    combined_deaths = []
    combined_kp = []
    for match_id, pair in by_match.items():
        user = pair.get("user")
        partner = pair.get("partner")
        if user is None or partner is None:
            continue
        team_id = int(_num(user.get("team_id")))
        team = context = team_context.get((match_id, team_id), {})
        team_damage = context.get("team_damage")
        team_taken = team.get("team_damage_taken")
        team_kills = team.get("team_kills")
        combined_damage_shares.append(
            _ratio(_num(user.get("damage_to_champions")) + _num(partner.get("damage_to_champions")), team_damage)
        )
        combined_taken_shares.append(
            _ratio(_num(user.get("damage_taken")) + _num(partner.get("damage_taken")), team_taken)
        )
        combined_deaths.append(_num(user.get("deaths")) + _num(partner.get("deaths")))
        combined_kp.append(
            _ratio(
                _num(user.get("kills")) + _num(user.get("assists")) + _num(partner.get("kills")) + _num(partner.get("assists")),
                team_kills * 2 if team_kills else None,
            )
        )
    return {
        "avg_combined_damage_share": _round_mean(combined_damage_shares),
        "avg_combined_damage_taken_share": _round_mean(combined_taken_shares),
        "avg_combined_deaths": _round_mean(combined_deaths),
        "avg_combined_kill_participation": _round_mean(combined_kp),
    }


def _style_signals(rows: Any) -> dict[str, Any]:
    if rows.empty:
        return {}
    return {
        "damage_share": _bucket(_mean(rows, "damage_share"), high=0.28, low=0.18),
        "damage_taken_share": _bucket(_mean(rows, "damage_taken_share"), high=0.28, low=0.16),
        "death_rate": _bucket(_per_10(rows, "deaths"), high=6.2, low=4.0),
        "cc_time": _bucket(_mean(rows, "time_ccing_others"), high=28, low=12),
        "mitigation": _bucket(_mean(rows, "damage_self_mitigated"), high=25000, low=10000),
        "kill_participation": _bucket(_mean(rows, "kill_participation"), high=0.72, low=0.55),
    }


def _recent_match_summaries(matches: Any, self_rows: Any) -> list[dict[str, Any]]:
    rows = []
    by_match = {str(row.get("match_id")): row for _, row in self_rows.iterrows()}
    for _, match in matches.head(12).iterrows():
        match_id = str(match.get("match_id"))
        user = by_match.get(match_id)
        if user is None:
            continue
        rows.append(
            {
                "match_id": match_id,
                "created": _clean(match.get("game_creation")),
                "win": _truthy(match.get("win")),
                "champion": _clean(user.get("champion_name")),
                "kda": _clean(user.get("kda")),
                "kills": _clean(user.get("kills")),
                "deaths": _clean(user.get("deaths")),
                "assists": _clean(user.get("assists")),
                "damage": _clean(user.get("damage_to_champions")),
                "damage_taken": _clean(user.get("damage_taken")),
            }
        )
    return rows


def _recent_player_match_summaries(rows: Any, limit: int = 10) -> list[dict[str, Any]]:
    if rows.empty:
        return []
    recent = _sort_recent(rows).head(limit)
    output = []
    for _, row in recent.iterrows():
        output.append(
            {
                "match_id": str(row.get("match_id")),
                "created": _clean(row.get("game_creation")),
                "win": _truthy(row.get("win")),
                "champion": _clean(row.get("champion_name")),
                "kda": _clean(row.get("kda")),
                "kills": _clean(row.get("kills")),
                "deaths": _clean(row.get("deaths")),
                "assists": _clean(row.get("assists")),
                "damage": _clean(row.get("damage_to_champions")),
                "damage_share": _clean(row.get("damage_share")),
                "damage_taken": _clean(row.get("damage_taken")),
                "damage_taken_share": _clean(row.get("damage_taken_share")),
                "kill_participation": _clean(row.get("kill_participation")),
            }
        )
    return output


def _top_champions(rows: Any, limit: int = 12) -> list[dict[str, Any]]:
    if rows.empty or "champion_name" not in rows.columns:
        return []
    output = []
    for champion, group in rows.groupby("champion_name"):
        output.append(
            {
                "champion": _clean(champion),
                "games": int(len(group)),
                "winrate": _ratio(_bool_sum(group, "win"), len(group)),
                "avg_kda": _mean(group, "kda"),
                "avg_damage": _mean(group, "damage_to_champions"),
                "avg_deaths": _mean(group, "deaths"),
            }
        )
    return sorted(output, key=lambda row: (row["games"], row["winrate"] or 0), reverse=True)[:limit]


def _top_pairs(rows: Any, columns: list[str], limit: int = 8) -> list[dict[str, Any]]:
    if rows.empty:
        return []
    counter = Counter()
    for _, row in rows.iterrows():
        values = tuple(str(int(_num(row.get(column)))) for column in columns if _num(row.get(column)))
        if values:
            counter[values] += 1
    return [{"ids": list(ids), "games": count} for ids, count in counter.most_common(limit)]


def _top_items(rows: Any, limit: int = 16) -> list[dict[str, Any]]:
    counter = Counter()
    for _, row in rows.iterrows():
        for index in range(7):
            item_id = int(_num(row.get(f"item{index}")))
            if item_id:
                counter[str(item_id)] += 1
    return [{"item_id": item, "count": count} for item, count in counter.most_common(limit)]


def _partner_key(row: Any) -> str:
    for key in ("puuid", "summoner_id", "riot_id", "summoner_name"):
        value = _clean(row.get(key))
        if value:
            return f"{key}:{value}"
    return f"participant:{row.get('match_id')}:{row.get('participant_id')}"


def _mean(rows: Any, column: str) -> float | None:
    if column not in rows.columns or rows.empty:
        return None
    values = [_num(value) for value in rows[column]]
    return _round_mean(values)


def _round_mean(values: list[float]) -> float | None:
    clean = [value for value in values if value is not None]
    return round(mean(clean), 4) if clean else None


def _per_min(rows: Any, column: str) -> float | None:
    if rows.empty or column not in rows.columns:
        return None
    values = []
    for _, row in rows.iterrows():
        duration = _num(row.get("duration_minutes"))
        if duration:
            values.append(_num(row.get(column)) / duration)
    return _round_mean(values)


def _per_10(rows: Any, column: str) -> float | None:
    value = _per_min(rows, column)
    return round(value * 10, 4) if value is not None else None


def _sum(rows: Any, column: str) -> float:
    if column not in rows.columns:
        return 0.0
    return float(sum(_num(value) for value in rows[column]))


def _bool_sum(rows: Any, column: str) -> int:
    if column not in rows.columns:
        return 0
    return sum(1 for value in rows[column] if _truthy(value))


def _ratio(numerator: float | int | None, denominator: float | int | None) -> float | None:
    if numerator is None or denominator in (None, 0):
        return None
    return round(float(numerator) / float(denominator), 4)


def _bucket(value: float | None, high: float, low: float) -> dict[str, Any]:
    if value is None:
        return {"value": None, "level": "unknown"}
    if value >= high:
        level = "high"
    elif value <= low:
        level = "low"
    else:
        level = "medium"
    return {"value": value, "level": level}


def _num(value: Any) -> float:
    if value is None:
        return 0.0
    try:
        if value != value:
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "win", "won", "victory", "胜"}
    return False


def _clean(value: Any) -> Any:
    if value is None:
        return ""
    if hasattr(value, "item"):
        try:
            return _clean(value.item())
        except (TypeError, ValueError):
            pass
    try:
        if value != value:
            return ""
    except TypeError:
        pass
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        return value.strip()
    return value


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat(timespec="seconds")
    if isinstance(value, dict):
        return {str(_json_safe(key)): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except (TypeError, ValueError):
            pass
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except (TypeError, ValueError):
            pass
    try:
        if value != value:
            return None
    except (TypeError, ValueError):
        pass
    return str(value)
