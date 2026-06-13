from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import requests

from llm_analysis import (
    LLMAnalysisError,
    build_analysis_payload,
    build_llm_context_export,
    build_llm_user_input,
    list_analysis_skills,
    load_combined_skill_prompt,
    load_local_env,
    load_skill_prompt,
    _post_json,
    _resolve_max_output_tokens,
    _resolve_timeout_seconds,
    _with_truncation_notice,
)
from storage import participants_csv_path, matches_csv_path, write_matches_csv, write_rows_csv


class LLMAnalysisTests(unittest.TestCase):
    def test_load_default_skill_prompt(self) -> None:
        prompt = load_skill_prompt()

        self.assertIn("车队复盘分析师", prompt)
        self.assertIn("车队", prompt)

    def test_load_combined_skill_prompt_combines_multiple_skills(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            first = Path(tmp) / "one" / "SKILL.md"
            second = Path(tmp) / "two" / "SKILL.md"
            first.parent.mkdir()
            second.parent.mkdir()
            first.write_text("---\nname: one\n---\nFirst body", encoding="utf-8")
            second.write_text("---\nname: two\n---\nSecond body", encoding="utf-8")

            prompt = load_combined_skill_prompt([first, second])

        self.assertIn("Skill 1: one", prompt)
        self.assertIn("First body", prompt)
        self.assertIn("Skill 2: two", prompt)
        self.assertIn("Second body", prompt)

    def test_list_analysis_skills_includes_default_and_user_skill(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            default = root / "skills" / "aram-match-analyst"
            default.mkdir(parents=True)
            default.joinpath("SKILL.md").write_text(
                "---\nname: default-skill\ndescription: Default\n---\nBody",
                encoding="utf-8",
            )
            custom = root / "skills" / "user" / "custom"
            custom.mkdir(parents=True)
            custom.joinpath("SKILL.md").write_text(
                "---\nname: custom-skill\ndescription: Custom\n---\nBody",
                encoding="utf-8",
            )

            names = [skill.name for skill in list_analysis_skills(root)]

        self.assertEqual(["default-skill", "custom-skill"], names)

    def test_load_local_env_reads_dotenv_without_overriding_existing_env(self) -> None:
        import os

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            root.joinpath(".env").write_text(
                "OPENAI_API_KEY=from-file\nLLM_MODEL=\"model-from-file\"\n",
                encoding="utf-8",
            )
            old_key = os.environ.get("OPENAI_API_KEY")
            old_model = os.environ.get("LLM_MODEL")
            try:
                os.environ["OPENAI_API_KEY"] = "already-set"
                os.environ.pop("LLM_MODEL", None)

                loaded = load_local_env(root)

                self.assertEqual("already-set", os.environ["OPENAI_API_KEY"])
                self.assertEqual("model-from-file", os.environ["LLM_MODEL"])
                self.assertIn("LLM_MODEL", loaded)
            finally:
                if old_key is None:
                    os.environ.pop("OPENAI_API_KEY", None)
                else:
                    os.environ["OPENAI_API_KEY"] = old_key
                if old_model is None:
                    os.environ.pop("LLM_MODEL", None)
                else:
                    os.environ["LLM_MODEL"] = old_model

    def test_build_analysis_payload_detects_frequent_ally(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp)
            write_matches_csv(
                matches_csv_path(data),
                [
                    {"match_id": "1", "game_creation_ms": 2, "win": True},
                    {"match_id": "2", "game_creation_ms": 1, "win": False},
                ],
            )
            rows = []
            for match_id, win in [("1", True), ("2", False)]:
                rows.extend(
                    [
                        {
                            "match_id": match_id,
                            "side": "self",
                            "team_id": 100,
                            "riot_id": "Me#CN1",
                            "summoner_id": "me",
                            "champion_name": "A",
                            "win": win,
                            "kills": 10,
                            "deaths": 5,
                            "assists": 20,
                            "kda": 6,
                            "duration_minutes": 20,
                            "damage_to_champions": 30000,
                            "damage_taken": 25000,
                            "gold_earned": 15000,
                        },
                        {
                            "match_id": match_id,
                            "side": "ally",
                            "team_id": 100,
                            "riot_id": "Ally#CN1",
                            "summoner_id": "ally",
                            "champion_name": "B",
                            "win": win,
                            "kills": 8,
                            "deaths": 6,
                            "assists": 18,
                            "kda": 4.33,
                            "duration_minutes": 20,
                            "damage_to_champions": 26000,
                            "damage_taken": 18000,
                            "gold_earned": 13000,
                        },
                        {
                            "match_id": match_id,
                            "side": "enemy",
                            "team_id": 200,
                            "riot_id": "Enemy#CN1",
                            "summoner_id": "enemy",
                            "champion_name": "C",
                            "win": not win,
                            "kills": 6,
                            "deaths": 8,
                            "assists": 12,
                            "kda": 2.25,
                            "duration_minutes": 20,
                            "damage_to_champions": 22000,
                            "damage_taken": 20000,
                            "gold_earned": 12000,
                        },
                    ]
                )
            write_rows_csv(participants_csv_path(data), rows)

            payload = build_analysis_payload(data, min_partner_games=2)

        self.assertEqual(2, payload["metadata"]["match_count"])
        self.assertEqual("Me#CN1", payload["user"]["identity"]["riot_id"])
        self.assertEqual(1, len(payload["frequent_allies"]))
        self.assertEqual("Ally#CN1", payload["frequent_allies"][0]["partner"]["riot_id"])
        self.assertEqual(2, len(payload["players_for_equal_analysis"]))
        self.assertEqual("self", payload["players_for_equal_analysis"][0]["role"])
        self.assertEqual(2, len(payload["recent_performance_ranking_seed"]))

    def test_build_analysis_payload_filters_selected_match_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp)
            write_matches_csv(
                matches_csv_path(data),
                [
                    {"match_id": "1", "game_creation_ms": 3, "win": True},
                    {"match_id": "2", "game_creation_ms": 2, "win": False},
                    {"match_id": "3", "game_creation_ms": 1, "win": True},
                ],
            )
            rows = []
            for match_id in ["1", "2", "3"]:
                rows.extend(
                    [
                        {
                            "match_id": match_id,
                            "side": "self",
                            "team_id": 100,
                            "riot_id": "Me#CN1",
                            "summoner_id": "me",
                            "champion_name": "A",
                            "win": match_id != "2",
                            "kills": 10,
                            "deaths": 5,
                            "assists": 20,
                            "duration_minutes": 20,
                            "damage_to_champions": 30000,
                            "damage_taken": 25000,
                        },
                        {
                            "match_id": match_id,
                            "side": "enemy",
                            "team_id": 200,
                            "riot_id": "Enemy#CN1",
                            "summoner_id": "enemy",
                            "champion_name": "C",
                            "win": match_id == "2",
                            "kills": 6,
                            "deaths": 8,
                            "assists": 12,
                            "duration_minutes": 20,
                            "damage_to_champions": 22000,
                            "damage_taken": 20000,
                        },
                    ]
                )
            write_rows_csv(participants_csv_path(data), rows)

            payload = build_analysis_payload(data, recent_games=3, match_ids=["2", "3"])

        self.assertEqual("explicit_match_ids", payload["metadata"]["match_selection"])
        self.assertEqual(["2", "3"], payload["metadata"]["selected_match_ids"])
        self.assertEqual(2, payload["metadata"]["match_count"])
        self.assertEqual(4, payload["metadata"]["participant_count"])

    def test_build_analysis_payload_includes_named_squad_member_below_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp)
            cache = data / "cache"
            cache.mkdir()
            cache.joinpath("ddragon_zh_CN.json").write_text(
                """
{
  "version": "test",
  "champions": {},
  "items": {
    "3083": {"name": "狂徒铠甲"},
    "3158": {"name": "明朗之靴"}
  },
  "summoner_spells": {
    "4": {"key": "4", "name": "闪现"},
    "32": {"key": "32", "name": "标记"}
  }
}
""",
                encoding="utf-8",
            )
            write_matches_csv(
                matches_csv_path(data),
                [{"match_id": "1", "game_creation_ms": 1, "win": True}],
            )
            write_rows_csv(
                participants_csv_path(data),
                [
                    {
                        "match_id": "1",
                        "side": "self",
                        "team_id": 100,
                        "riot_id": "Me#CN1",
                        "summoner_id": "me",
                        "champion_name": "A",
                        "win": True,
                        "kills": 10,
                        "deaths": 4,
                        "assists": 20,
                        "kda": 7.5,
                        "duration_minutes": 20,
                        "damage_to_champions": 30000,
                        "damage_taken": 18000,
                        "damage_self_mitigated": 8000,
                        "total_heal": 1000,
                        "time_ccing_others": 10,
                        "gold_earned": 15000,
                        "spell1_id": 4,
                        "spell2_id": 32,
                    },
                    {
                        "match_id": "1",
                        "side": "ally",
                        "team_id": 100,
                        "riot_id": "tbc02#CN1",
                        "summoner_id": "tbc02-id",
                        "champion_name": "Malphite",
                        "win": True,
                        "kills": 6,
                        "deaths": 8,
                        "assists": 25,
                        "kda": 3.875,
                        "duration_minutes": 20,
                        "damage_to_champions": 18000,
                        "damage_taken": 42000,
                        "damage_self_mitigated": 32000,
                        "total_heal": 500,
                        "time_ccing_others": 35,
                        "gold_earned": 12000,
                        "spell1_id": 4,
                        "spell2_id": 32,
                        "item0": 3083,
                        "item1": 3158,
                    },
                    {
                        "match_id": "1",
                        "side": "enemy",
                        "team_id": 200,
                        "riot_id": "Enemy#CN1",
                        "summoner_id": "enemy",
                        "champion_name": "C",
                        "win": False,
                        "kills": 2,
                        "deaths": 10,
                        "assists": 8,
                        "kda": 1,
                        "duration_minutes": 20,
                        "damage_to_champions": 12000,
                        "damage_taken": 20000,
                    },
                ],
            )

            payload = build_analysis_payload(
                data,
                min_partner_games=2,
                squad_member_names="tbc02,tbc05",
            )

        profiles = payload["players_for_equal_analysis"]
        tbc02_profile = next(profile for profile in profiles if profile["identity"]["riot_id"] == "tbc02#CN1")
        self.assertEqual("squad_member", tbc02_profile["role"])
        self.assertTrue(tbc02_profile["is_named_squad_member"])
        self.assertEqual(["tbc02"], payload["metadata"]["detected_squad_members"])
        self.assertEqual("named_squad_member", payload["frequent_allies"][0]["partner_type"])
        self.assertIn("function_mix", tbc02_profile)
        self.assertEqual("前排开团", tbc02_profile["champion_function_profiles"][0]["observed_function_zh"])
        self.assertEqual("狂徒铠甲", tbc02_profile["items"][0]["item_name"])
        self.assertEqual(["闪现", "标记"], tbc02_profile["spells"][0]["names"])

    def test_build_llm_user_input_serializes_pandas_numpy_scalars(self) -> None:
        payload = {
            "value": pd.Series([1], dtype="int64").iloc[0],
            "missing": float("nan"),
        }

        text = build_llm_user_input(payload)

        self.assertIn('"value": 1', text)
        self.assertIn('"missing": null', text)

    def test_build_llm_context_export_includes_responses_request_body(self) -> None:
        export = build_llm_context_export(
            payload={"value": pd.Series([1], dtype="int64").iloc[0]},
            system_prompt="system text",
            model="test-model",
            base_url="https://example.test/v1",
            api_style="responses",
            max_output_tokens=1234,
            reasoning_effort="low",
            timeout=99,
            skill_paths=[Path("skills/example/SKILL.md")],
        )

        self.assertEqual("responses", export["metadata"]["api_style"])
        self.assertEqual("https://example.test/v1/responses", export["metadata"]["api_url"])
        self.assertEqual(["skills/example/SKILL.md"], export["metadata"]["skill_paths"])
        self.assertEqual("system text", export["system_prompt"])
        self.assertIn('"value": 1', export["user_input"])
        self.assertEqual("system text", export["request_body"]["instructions"])
        self.assertEqual(export["user_input"], export["request_body"]["input"])
        self.assertEqual({"effort": "low"}, export["request_body"]["reasoning"])
        self.assertNotIn("api_key", json.dumps(export, ensure_ascii=False).lower())
        self.assertNotIn("authorization", json.dumps(export, ensure_ascii=False).lower())

    def test_build_llm_context_export_includes_chat_messages(self) -> None:
        export = build_llm_context_export(
            payload={"value": "ok"},
            system_prompt="system text",
            model="test-model",
            api_style="chat",
            max_output_tokens=1234,
        )

        messages = export["request_body"]["messages"]
        self.assertEqual("chat", export["metadata"]["api_style"])
        self.assertEqual("system", messages[0]["role"])
        self.assertEqual("system text", messages[0]["content"])
        self.assertEqual("user", messages[1]["role"])
        self.assertEqual(export["user_input"], messages[1]["content"])
        self.assertEqual(1234, export["request_body"]["max_tokens"])

    def test_build_llm_context_export_includes_deepseek_chat_body(self) -> None:
        export = build_llm_context_export(
            payload={"value": "ok"},
            system_prompt="system text",
            api_style="deepseek",
            max_output_tokens=1234,
        )

        self.assertEqual("deepseek", export["metadata"]["provider"])
        self.assertEqual("chat", export["metadata"]["api_style"])
        self.assertEqual("https://api.deepseek.com/chat/completions", export["metadata"]["api_url"])
        self.assertEqual("deepseek-chat", export["request_body"]["model"])
        self.assertEqual(export["user_input"], export["request_body"]["messages"][1]["content"])

    def test_build_llm_context_export_includes_anthropic_messages_body(self) -> None:
        export = build_llm_context_export(
            payload={"value": "ok"},
            system_prompt="system text",
            model="claude-test",
            api_style="anthropic",
            max_output_tokens=1234,
        )

        body = export["request_body"]
        self.assertEqual("anthropic", export["metadata"]["provider"])
        self.assertEqual("anthropic", export["metadata"]["api_style"])
        self.assertEqual("https://api.anthropic.com/v1/messages", export["metadata"]["api_url"])
        self.assertEqual("system text", body["system"])
        self.assertEqual("user", body["messages"][0]["role"])
        self.assertEqual(export["user_input"], body["messages"][0]["content"])
        self.assertEqual(1234, body["max_tokens"])

    def test_resolve_max_output_tokens_reads_env(self) -> None:
        import os

        old_value = os.environ.get("LLM_MAX_OUTPUT_TOKENS")
        try:
            os.environ["LLM_MAX_OUTPUT_TOKENS"] = "12000"

            self.assertEqual(12000, _resolve_max_output_tokens(None))
            self.assertEqual(3000, _resolve_max_output_tokens(3000))
        finally:
            if old_value is None:
                os.environ.pop("LLM_MAX_OUTPUT_TOKENS", None)
            else:
                os.environ["LLM_MAX_OUTPUT_TOKENS"] = old_value

    def test_resolve_timeout_seconds_reads_env(self) -> None:
        import os

        old_value = os.environ.get("LLM_TIMEOUT_SECONDS")
        try:
            os.environ["LLM_TIMEOUT_SECONDS"] = "900"

            self.assertEqual(900, _resolve_timeout_seconds(None))
            self.assertEqual(120, _resolve_timeout_seconds(120))
        finally:
            if old_value is None:
                os.environ.pop("LLM_TIMEOUT_SECONDS", None)
            else:
                os.environ["LLM_TIMEOUT_SECONDS"] = old_value

    def test_post_json_wraps_timeout(self) -> None:
        with patch("llm_analysis.requests.post", side_effect=requests.ReadTimeout("slow")):
            with self.assertRaisesRegex(LLMAnalysisError, "timed out"):
                _post_json(
                    url="https://api.openai.com/v1/responses",
                    api_key="test-key",
                    body={"model": "test"},
                    timeout=1,
                )

    def test_truncation_notice_is_prepended_to_incomplete_response(self) -> None:
        text = _with_truncation_notice(
            "Report body",
            {
                "status": "incomplete",
                "incomplete_details": {"reason": "max_output_tokens"},
            },
        )

        self.assertIn("max_output_tokens", text)

    def test_truncation_notice_handles_anthropic_max_tokens(self) -> None:
        text = _with_truncation_notice("Report body", {"stop_reason": "max_tokens"})

        self.assertIn("stop_reason=max_tokens", text)
        self.assertIn("LLM_MAX_OUTPUT_TOKENS", text)
        self.assertTrue(text.endswith("Report body"))


if __name__ == "__main__":
    unittest.main()
