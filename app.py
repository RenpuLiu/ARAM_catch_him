from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from datadragon import load_static_maps
from llm_analysis import (
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_SQUAD_MEMBER_NAMES,
    DEFAULT_TIMEOUT_SECONDS,
    LLMAnalysisError,
    build_analysis_payload,
    build_llm_context_export,
    call_llm,
    list_analysis_skills,
    load_combined_skill_prompt,
    load_local_env,
    save_report,
)
from lol_cn_lcu_dump import dump_lcu_data
from match_parser import is_aram_game
from stats import build_summary, champion_stats, format_percent, hour_stats, item_stats
from storage import matches_csv_path, participants_csv_path, summary_path


st.set_page_config(page_title="ARAM Catch Him", layout="wide")

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.35rem; padding-bottom: 2.4rem; }
    div[data-testid="stMetric"] {
        border: 1px solid rgba(49, 51, 63, 0.16);
        border-radius: 6px;
        padding: 0.72rem 0.86rem;
        background: #fbfbfd;
    }
    div[data-testid="stMetricValue"] { font-size: 1.55rem; }
    h1, h2, h3 { letter-spacing: 0; }
    .match-caption {
        color: #5f6572;
        font-size: 0.92rem;
        margin: -0.15rem 0 0.45rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


COMPACT_PLAYER_COLUMNS = [
    "身份",
    "玩家",
    "英雄",
    "K/D/A",
    "KDA",
    "等级",
    "输出",
    "承伤",
    "金币",
    "补刀",
    "技能",
    "装备",
]

DETAIL_PLAYER_COLUMNS = [
    "身份",
    "队伍",
    "玩家",
    "英雄",
    "胜负",
    "K/D/A",
    "KDA",
    "等级",
    "输出",
    "物理伤害",
    "魔法伤害",
    "真实伤害",
    "承伤",
    "物理承伤",
    "魔法承伤",
    "真实承伤",
    "自我减伤",
    "治疗",
    "控制",
    "金币",
    "花费",
    "补刀",
    "多杀",
    "视野",
    "技能",
    "装备",
    "装备ID",
    "符文ID",
    "增强ID",
    "match_id",
]


def main() -> None:
    st.title("ARAM Catch Him")

    with st.sidebar:
        st.header("数据")
        data_dir = st.text_input("目录", value="data")
        count = st.number_input("最近对局", min_value=1, max_value=200, value=50, step=5)
        lockfile = st.text_input("lockfile", value="")
        include_timelines = st.checkbox("抓取 timeline", value=False)
        refresh_static = st.checkbox("刷新静态映射", value=False)

        if st.button("从 LCU 更新", use_container_width=True):
            with st.spinner("正在读取本机客户端..."):
                try:
                    result = dump_lcu_data(
                        n=int(count),
                        out=data_dir,
                        lockfile=lockfile.strip() or None,
                        include_timelines=include_timelines,
                    )
                    st.success(
                        f"本次抓取 {result.get('fetched_matches', 0)} 局，"
                        f"新增 {result.get('added_matches', 0)} 局，"
                        f"更新/去重 {result.get('updated_matches', 0)} 局；"
                        f"历史共 {result.get('total_matches', result['normalized_matches'])} 局，"
                        f"{result.get('total_participants', result['normalized_participants'])} 条玩家记录。"
                    )
                    if result["errors"]:
                        st.warning(
                            f"{len(result['errors'])} 个详情请求失败，已写入 raw/fetch_errors.json。"
                        )
                except Exception as exc:
                    st.error(str(exc))

    rows = _load_rows(data_dir)
    if rows.empty:
        _empty_state(data_dir)
        return

    rows = _enrich_champion_names(rows, data_dir, refresh_static)
    participants = _enrich_participants(_load_participants(data_dir), data_dir, refresh_static)

    aram_only = st.toggle("只看 ARAM", value=True)
    max_games = st.slider("展示局数", min_value=5, max_value=200, value=min(50, len(rows)), step=5)

    filtered = _filtered(rows, aram_only).head(max_games)
    records = filtered.to_dict("records")
    summary = build_summary(records)
    participant_rows = _participants_for_matches(participants, filtered)

    metric_cols = st.columns(6)
    metric_cols[0].metric("场次", summary["games"])
    metric_cols[1].metric("胜率", format_percent(summary["winrate"]))
    metric_cols[2].metric("近 20 胜率", format_percent(summary["recent_20_winrate"]))
    metric_cols[3].metric("KDA", _fmt(summary["avg_kda"]))
    metric_cols[4].metric("输出", _fmt(summary["avg_damage_to_champions"], digits=0))
    metric_cols[5].metric("承伤", _fmt(summary["avg_damage_taken"], digits=0))

    tab_recent, tab_players, tab_llm, tab_champs, tab_time, tab_items, tab_raw = st.tabs(
        ["最近对局", "阵容", "LLM 分析", "英雄池", "时间段", "装备", "文件"]
    )

    with tab_recent:
        st.dataframe(
            _recent_view(filtered),
            use_container_width=True,
            hide_index=True,
            height=520,
        )

    with tab_players:
        _players_tab(filtered, participant_rows)

    with tab_llm:
        _llm_tab(data_dir)

    with tab_champs:
        champion_df = pd.DataFrame(champion_stats(records))
        if not champion_df.empty:
            champion_df["winrate"] = champion_df["winrate"].map(format_percent)
            st.dataframe(champion_df, use_container_width=True, hide_index=True, height=520)
            st.bar_chart(champion_df.head(12).set_index("champion")["games"])

    with tab_time:
        hour_df = pd.DataFrame(hour_stats(records))
        if not hour_df.empty:
            display = hour_df.copy()
            display["winrate"] = display["winrate"].map(format_percent)
            st.dataframe(display, use_container_width=True, hide_index=True, height=420)
            st.bar_chart(hour_df.set_index("hour")[["games"]])

    with tab_items:
        items_df = pd.DataFrame(item_stats(records))
        if not items_df.empty:
            items_df["item_name"] = items_df["item_id"].map(
                lambda value: _item_name(value, data_dir, refresh_static)
            )
            items_df["winrate"] = items_df["winrate"].map(format_percent)
            st.dataframe(
                items_df[["item_id", "item_name", "games", "wins", "winrate"]],
                use_container_width=True,
                hide_index=True,
                height=520,
            )

    with tab_raw:
        st.json(_load_summary(data_dir))
        st.caption(str(Path(data_dir).resolve()))


def _llm_tab(data_dir: str) -> None:
    load_local_env()
    skills = list_analysis_skills()
    if not skills:
        st.error("没有找到分析 skill。请确认 skills/aram-match-analyst/SKILL.md 存在。")
        return

    skill_labels = [
        f"{skill.name} · {skill.path}"
        for skill in skills
    ]
    skill_by_label = dict(zip(skill_labels, skills))

    config_cols = st.columns([2, 1, 1])
    with config_cols[0]:
        selected_labels = st.multiselect(
            "分析 skills",
            options=skill_labels,
            default=skill_labels[:1],
            help="可多选；多个 SKILL.md 会按选择顺序合并为一个 system prompt。",
        )
    with config_cols[1]:
        recent_games = st.number_input("分析局数", min_value=5, max_value=200, value=50, step=5)
    with config_cols[2]:
        min_partner_games = st.number_input("常见队友阈值", min_value=1, max_value=20, value=2, step=1)

    squad_members_text = st.text_input(
        "车队成员",
        value=os.getenv("LLM_SQUAD_MEMBERS", ", ".join(DEFAULT_SQUAD_MEMBER_NAMES)),
        help="逗号分隔；这些成员只要出现在数据里，就会进入同维度分析。",
    )

    api_cols = st.columns([1, 1, 1])
    with api_cols[0]:
        model = st.text_input("模型", value=os.getenv("LLM_MODEL", "gpt-5.4"))
    with api_cols[1]:
        api_style = st.selectbox(
            "接口",
            options=["responses", "chat"],
            index=0 if os.getenv("LLM_API_STYLE", "responses") != "chat" else 1,
        )
    with api_cols[2]:
        base_url = st.text_input("Base URL", value=os.getenv("LLM_BASE_URL", "https://api.openai.com/v1"))

    generation_cols = st.columns([1, 1, 1])
    with generation_cols[0]:
        max_output_tokens = st.number_input(
            "输出 token 上限",
            min_value=1000,
            max_value=32000,
            value=_env_positive_int("LLM_MAX_OUTPUT_TOKENS", DEFAULT_MAX_OUTPUT_TOKENS),
            step=500,
            help="报告被截断时调高这个值。",
        )
    with generation_cols[1]:
        timeout_seconds = st.number_input(
            "API 超时秒数",
            min_value=60,
            max_value=1800,
            value=_env_positive_int("LLM_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS),
            step=60,
            help="长报告生成较慢时调高这个值。",
        )
    with generation_cols[2]:
        reasoning_options = _reasoning_effort_options()
        reasoning_effort = st.selectbox(
            "推理强度",
            options=reasoning_options,
            index=reasoning_options.index(_default_reasoning_effort(reasoning_options)),
            help="Responses API 可用；default/none 不会主动传 reasoning 参数。",
        )

    api_key = st.text_input(
        "API Key",
        value="",
        type="password",
        placeholder="留空则读取 OPENAI_API_KEY 或 LLM_API_KEY",
    )

    selected_skills = [skill_by_label[label] for label in selected_labels]
    if not selected_skills:
        st.warning("请至少选择一个分析 skill。")
        return
    try:
        payload = build_analysis_payload(
            data_dir=data_dir,
            min_partner_games=int(min_partner_games),
            recent_games=int(recent_games),
            squad_member_names=squad_members_text,
        )
    except LLMAnalysisError as exc:
        st.error(str(exc))
        return

    meta = payload["metadata"]
    st.caption(
        f"将分析 {meta['match_count']} 局、{meta['participant_count']} 条玩家记录；"
        f"识别到 {len(payload['frequent_allies'])} 个常见队友/疑似多排；"
        f"同维度玩家分析 {len(payload['players_for_equal_analysis'])} 人；"
        f"命中车队成员 {len(payload['metadata'].get('detected_squad_members', []))} 人。"
    )

    preview_cols = st.columns(2)
    with preview_cols[0]:
        st.markdown("**用户平均表现摘要**")
        st.json(payload["user"]["overall"], expanded=False)
    with preview_cols[1]:
        st.markdown("**最近表现排序参考**")
        st.json(payload["recent_performance_ranking_seed"][:8], expanded=False)

    try:
        system_prompt = load_combined_skill_prompt([skill.path for skill in selected_skills])
        context_export = build_llm_context_export(
            payload=payload,
            system_prompt=system_prompt,
            model=model.strip() or None,
            base_url=base_url.strip() or None,
            api_style=api_style,
            max_output_tokens=int(max_output_tokens),
            reasoning_effort=None if reasoning_effort == "default" else reasoning_effort,
            timeout=int(timeout_seconds),
            skill_paths=[skill.path for skill in selected_skills],
        )
    except LLMAnalysisError as exc:
        st.error(str(exc))
        return

    st.download_button(
        "下载完整模型上下文 JSON",
        data=json.dumps(context_export, ensure_ascii=False, indent=2),
        file_name=_llm_context_filename(),
        mime="application/json",
        use_container_width=True,
    )

    with st.expander("查看将发送给模型的完整上下文", expanded=False):
        st.json(context_export, expanded=False)

    if st.button("生成 LLM 分析报告", use_container_width=True):
        effective_key = api_key.strip() or None
        if not effective_key and not (os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")):
            st.warning("请填写 API Key，或在环境变量中设置 OPENAI_API_KEY / LLM_API_KEY。")
            return

        with st.spinner("正在生成分析报告..."):
            try:
                user_input = context_export["user_input"]
                report, raw_response = call_llm(
                    system_prompt=system_prompt,
                    user_input=user_input,
                    model=model.strip() or None,
                    api_key=effective_key,
                    base_url=base_url.strip() or None,
                    api_style=api_style,
                    max_output_tokens=int(max_output_tokens),
                    reasoning_effort=None if reasoning_effort == "default" else reasoning_effort,
                    timeout=int(timeout_seconds),
                )
                report_path = save_report(data_dir, report, payload, raw_response)
            except LLMAnalysisError as exc:
                st.error(str(exc))
                return

        st.success(f"报告已保存：{report_path}")
        st.markdown(report)


def _llm_context_filename() -> str:
    return f"llm_context_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"


def _load_rows(data_dir: str) -> pd.DataFrame:
    path = matches_csv_path(data_dir)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "game_creation_ms" in df.columns:
        df = df.sort_values("game_creation_ms", ascending=False)
    return df


def _load_participants(data_dir: str) -> pd.DataFrame:
    path = participants_csv_path(data_dir)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "game_creation_ms" in df.columns:
        df = df.sort_values("game_creation_ms", ascending=False)
    return df


def _filtered(rows: pd.DataFrame, aram_only: bool) -> pd.DataFrame:
    if not aram_only:
        return rows
    return rows[rows.apply(lambda row: _row_is_aram(row.to_dict()), axis=1)]


def _row_is_aram(row: dict[str, Any]) -> bool:
    return _truthy(row.get("is_aram")) or is_aram_game(row)


def _env_positive_int(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        return default
    return value if value > 0 else default


def _reasoning_effort_options() -> list[str]:
    options = ["default", "low", "medium", "high", "none"]
    current = (os.getenv("LLM_REASONING_EFFORT") or "").strip().lower()
    if current and current not in options:
        options.insert(1, current)
    return options


def _default_reasoning_effort(options: list[str]) -> str:
    current = (os.getenv("LLM_REASONING_EFFORT") or "").strip().lower()
    return current if current in options else "default"


def _participants_for_matches(participants: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    if participants.empty or matches.empty or "match_id" not in participants.columns:
        return pd.DataFrame()
    match_ids = set(matches["match_id"].astype(str))
    return participants[participants["match_id"].astype(str).isin(match_ids)]


def _enrich_champion_names(rows: pd.DataFrame, data_dir: str, refresh: bool) -> pd.DataFrame:
    if rows.empty or "champion_id" not in rows.columns:
        return rows
    try:
        maps = _static_maps(data_dir, refresh)
    except Exception:
        return rows

    enriched = rows.copy()
    champion_map = maps.get("champions", {})
    enriched["champion_name"] = enriched.apply(
        lambda row: champion_map.get(
            _safe_int(row.get("champion_id")),
            row.get("champion_name") or str(row.get("champion_id") or ""),
        ),
        axis=1,
    )
    return enriched


def _enrich_participants(rows: pd.DataFrame, data_dir: str, refresh: bool) -> pd.DataFrame:
    rows = _enrich_champion_names(rows, data_dir, refresh)
    if rows.empty:
        return rows

    try:
        maps = _static_maps(data_dir, refresh)
    except Exception:
        maps = {"items": {}, "summoner_spells": {}}

    enriched = rows.copy()
    item_map = maps.get("items", {})
    spell_map = maps.get("summoner_spells", {})
    enriched["装备"] = enriched.apply(lambda row: _item_names(row, item_map), axis=1)
    enriched["装备ID"] = enriched.apply(lambda row: _item_ids_text(row), axis=1)
    enriched["技能"] = enriched.apply(lambda row: _spell_names(row, spell_map), axis=1)
    enriched["符文ID"] = enriched.get("perks_json", pd.Series(dtype=str)).map(_json_list_text)
    enriched["增强ID"] = enriched.get("augments_json", pd.Series(dtype=str)).map(_json_list_text)
    return enriched


def _recent_view(rows: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "game_creation",
        "queue_id",
        "game_mode",
        "champion_name",
        "win",
        "kills",
        "deaths",
        "assists",
        "kda",
        "damage_to_champions",
        "damage_taken",
        "duration_minutes",
        "teammate_names",
        "opponent_names",
        "match_id",
    ]
    existing = [column for column in columns if column in rows.columns]
    view = rows[existing].copy()
    if "win" in view.columns:
        view["win"] = view["win"].map(lambda value: "胜" if _truthy(value) else "负")
    return view


def _players_tab(matches: pd.DataFrame, participants: pd.DataFrame) -> None:
    if participants.empty:
        st.info("刷新一次数据后会生成玩家明细。")
        return

    match_options = matches.reset_index(drop=True)
    if match_options.empty:
        st.info("当前过滤条件下没有对局。")
        return

    st.caption("每局单独分组；默认表保留复盘最常看的字段，详细字段里有装备、符文、伤害拆分、治疗、控制、视野等。")

    for index, match in match_options.iterrows():
        match_id = str(match.get("match_id", ""))
        selected_players = participants[participants["match_id"].astype(str) == match_id]
        if selected_players.empty:
            continue

        with st.expander(_match_label(match), expanded=index == 0):
            _match_header(match, selected_players)
            ally_players = _sort_players(
                selected_players[selected_players["side"].isin(["self", "ally"])]
            )
            enemy_players = _sort_players(selected_players[selected_players["side"] == "enemy"])

            team_cols = st.columns(2)
            with team_cols[0]:
                st.markdown("**我方**")
                st.dataframe(
                    _compact_player_view(ally_players),
                    use_container_width=True,
                    hide_index=True,
                    height=250,
                    column_config=_compact_column_config(),
                )
            with team_cols[1]:
                st.markdown("**对手**")
                st.dataframe(
                    _compact_player_view(enemy_players),
                    use_container_width=True,
                    hide_index=True,
                    height=250,
                    column_config=_compact_column_config(),
                )

            if st.checkbox("显示详细字段", key=f"detail_{match_id}"):
                st.dataframe(
                    _detail_player_view(selected_players),
                    use_container_width=True,
                    hide_index=True,
                    height=360,
                    column_config=_detail_column_config(),
                )

    with st.expander("所有已加载玩家记录", expanded=False):
        st.dataframe(
            _detail_player_view(participants),
            use_container_width=True,
            hide_index=True,
            height=560,
            column_config=_detail_column_config(),
        )


def _match_header(match: pd.Series, players: pd.DataFrame) -> None:
    result = "胜利" if _truthy(match.get("win")) else "失败"
    champion = match.get("champion_name") or match.get("champion_id") or ""
    duration = _fmt(match.get("duration_minutes"), digits=1)
    queue = match.get("queue_id") or ""
    damage = _team_total(players, "damage_to_champions")
    taken = _team_total(players, "damage_taken")
    st.markdown(
        f'<div class="match-caption">{result} · 我的英雄 {champion} · {duration} 分钟 · '
        f"queue {queue} · 两队总输出 {_fmt(damage, 0)} · 总承伤 {_fmt(taken, 0)}</div>",
        unsafe_allow_html=True,
    )


def _team_total(players: pd.DataFrame, column: str) -> float:
    if column not in players.columns:
        return 0.0
    return float(pd.to_numeric(players[column], errors="coerce").fillna(0).sum())


def _match_label(row: pd.Series) -> str:
    result = "胜" if _truthy(row.get("win")) else "负"
    champion = row.get("champion_name") or row.get("champion_id") or ""
    created = _short_time(row.get("game_creation"))
    match_id = row.get("match_id") or ""
    return f"{created} · {champion} · {result} · {match_id}"


def _compact_player_view(players: pd.DataFrame) -> pd.DataFrame:
    view = _base_player_view(players)
    existing = [column for column in COMPACT_PLAYER_COLUMNS if column in view.columns]
    return view[existing]


def _detail_player_view(players: pd.DataFrame) -> pd.DataFrame:
    view = _base_player_view(players)
    existing = [column for column in DETAIL_PLAYER_COLUMNS if column in view.columns]
    return view[existing]


def _base_player_view(players: pd.DataFrame) -> pd.DataFrame:
    if players.empty:
        return pd.DataFrame()

    view = _sort_players(players).copy()
    view["身份"] = view.get("side", "").map(
        {"self": "自己", "ally": "队友", "enemy": "对手", "unknown": "未知"}
    )
    view["队伍"] = view.get("team_id", "")
    view["玩家"] = view.apply(_player_name, axis=1)
    view["英雄"] = view.get("champion_name", "")
    view["胜负"] = view.get("win", "").map(lambda value: "胜" if _truthy(value) else "负")
    view["K/D/A"] = view.apply(
        lambda row: f"{_safe_int(row.get('kills'))}/{_safe_int(row.get('deaths'))}/{_safe_int(row.get('assists'))}",
        axis=1,
    )
    view["KDA"] = pd.to_numeric(view.get("kda"), errors="coerce")
    view["等级"] = _numeric_series(view, "champ_level")
    view["输出"] = _numeric_series(view, "damage_to_champions")
    view["物理伤害"] = _numeric_series(view, "physical_damage_to_champions")
    view["魔法伤害"] = _numeric_series(view, "magic_damage_to_champions")
    view["真实伤害"] = _numeric_series(view, "true_damage_to_champions")
    view["承伤"] = _numeric_series(view, "damage_taken")
    view["物理承伤"] = _numeric_series(view, "physical_damage_taken")
    view["魔法承伤"] = _numeric_series(view, "magic_damage_taken")
    view["真实承伤"] = _numeric_series(view, "true_damage_taken")
    view["自我减伤"] = _numeric_series(view, "damage_self_mitigated")
    view["治疗"] = _numeric_series(view, "total_heal")
    view["控制"] = _numeric_series(view, "time_ccing_others")
    view["金币"] = _numeric_series(view, "gold_earned")
    view["花费"] = _numeric_series(view, "gold_spent")
    view["补刀"] = _numeric_series(view, "cs")
    view["多杀"] = view.apply(_multi_kill_text, axis=1)
    view["视野"] = _numeric_series(view, "vision_score")
    return view


def _sort_players(players: pd.DataFrame) -> pd.DataFrame:
    if players.empty:
        return players
    view = players.copy()
    side_order = {"self": 0, "ally": 1, "enemy": 2, "unknown": 3}
    view["_side_order"] = view.get("side", "").map(side_order).fillna(9)
    sort_cols = [col for col in ["_side_order", "team_id", "participant_id"] if col in view.columns]
    return view.sort_values(sort_cols)


def _player_name(row: pd.Series) -> str:
    return (
        str(row.get("riot_id") or "").strip()
        or str(row.get("summoner_name") or "").strip()
        or "-"
    )


def _multi_kill_text(row: pd.Series) -> str:
    parts = []
    for label, key in [
        ("D", "double_kills"),
        ("T", "triple_kills"),
        ("Q", "quadra_kills"),
        ("P", "penta_kills"),
    ]:
        value = _safe_int(row.get(key))
        if value:
            parts.append(f"{label}{value}")
    return " ".join(parts) if parts else "-"


def _item_names(row: pd.Series, item_map: dict[int, str]) -> str:
    ids = _item_ids(row)
    names = [item_map.get(item_id, str(item_id)) for item_id in ids]
    return " / ".join(names) if names else "-"


def _item_ids_text(row: pd.Series) -> str:
    ids = _item_ids(row)
    return " / ".join(str(item_id) for item_id in ids) if ids else "-"


def _item_ids(row: pd.Series) -> list[int]:
    ids = []
    for key in [f"item{index}" for index in range(7)]:
        item_id = _safe_int(row.get(key))
        if item_id:
            ids.append(item_id)
    if ids:
        return ids
    return [_safe_int(value) for value in _json_list(row.get("items_json")) if _safe_int(value)]


def _spell_names(row: pd.Series, spell_map: dict[int, str]) -> str:
    ids = [_safe_int(row.get("spell1_id")), _safe_int(row.get("spell2_id"))]
    names = [spell_map.get(spell_id, str(spell_id)) for spell_id in ids if spell_id]
    return " / ".join(names) if names else "-"


def _json_list_text(value: Any) -> str:
    values = _json_list(value)
    return " / ".join(str(value) for value in values) if values else "-"


def _json_list(value: Any) -> list[Any]:
    if value is None or (isinstance(value, float) and value != value):
        return []
    if isinstance(value, list):
        return value
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError:
        return []
    return parsed if isinstance(parsed, list) else []


def _compact_column_config() -> dict[str, Any]:
    return {
        "玩家": st.column_config.TextColumn(width="medium"),
        "英雄": st.column_config.TextColumn(width="small"),
        "K/D/A": st.column_config.TextColumn(width="small"),
        "KDA": st.column_config.NumberColumn(format="%.2f", width="small"),
        "输出": st.column_config.NumberColumn(format="%d", width="small"),
        "承伤": st.column_config.NumberColumn(format="%d", width="small"),
        "金币": st.column_config.NumberColumn(format="%d", width="small"),
        "技能": st.column_config.TextColumn(width="medium"),
        "装备": st.column_config.TextColumn(width="large"),
    }


def _detail_column_config() -> dict[str, Any]:
    number_cols = {
        column: st.column_config.NumberColumn(format="%d", width="small")
        for column in [
            "输出",
            "物理伤害",
            "魔法伤害",
            "真实伤害",
            "承伤",
            "物理承伤",
            "魔法承伤",
            "真实承伤",
            "自我减伤",
            "治疗",
            "控制",
            "金币",
            "花费",
            "补刀",
            "视野",
        ]
    }
    number_cols["KDA"] = st.column_config.NumberColumn(format="%.2f", width="small")
    number_cols["装备"] = st.column_config.TextColumn(width="large")
    number_cols["装备ID"] = st.column_config.TextColumn(width="large")
    return number_cols


def _load_summary(data_dir: str) -> dict:
    path = summary_path(data_dir)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _item_name(item_id: int, data_dir: str, refresh: bool) -> str:
    try:
        maps = _static_maps(data_dir, refresh)
    except Exception:
        return str(item_id)
    return maps.get("items", {}).get(_safe_int(item_id), str(item_id))


@st.cache_data(show_spinner=False)
def _static_maps(data_dir: str, refresh: bool) -> dict:
    return load_static_maps(Path(data_dir) / "cache", refresh=refresh)


def _truthy(value: Any) -> bool:
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "win", "won", "victory", "胜"}
    return False


def _numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series([0] * len(df), index=df.index)
    return pd.to_numeric(df[column], errors="coerce").fillna(0)


def _safe_int(value: Any) -> int:
    if value is None:
        return 0
    try:
        if pd.isna(value):
            return 0
    except (TypeError, ValueError):
        pass
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "-"
    try:
        if pd.isna(value):
            return "-"
    except (TypeError, ValueError):
        pass
    return f"{float(value):,.{digits}f}"


def _short_time(value: Any) -> str:
    text = str(value or "")
    if "T" in text:
        date, time = text.split("T", 1)
        return f"{date} {time[:5]}"
    return text[:16]


def _empty_state(data_dir: str) -> None:
    st.info("还没有本地数据。登录英雄联盟客户端后，点侧边栏的“从 LCU 更新”。")
    st.code(
        f"python lol_cn_lcu_dump.py --n 50 --out {data_dir}\n"
        "streamlit run app.py",
        language="powershell",
    )


if __name__ == "__main__":
    main()
