from __future__ import annotations

import unittest

from match_parser import extract_games, is_aram_game, normalize_match, normalize_match_participants
from stats import build_summary, champion_stats, filter_rows


class MatchParserTests(unittest.TestCase):
    def test_extract_games_from_lcu_shape(self) -> None:
        payload = {"games": {"games": [{"gameId": 1}, {"gameId": 2}]}}
        self.assertEqual([1, 2], [game["gameId"] for game in extract_games(payload)])

    def test_normalize_match_uses_current_account_identity(self) -> None:
        game = {
            "gameId": 123,
            "queueId": 450,
            "gameCreation": 1710000000000,
            "gameDuration": 1300,
            "gameMode": "ARAM",
            "participants": [
                {
                    "participantId": 1,
                    "teamId": 100,
                    "championId": 22,
                    "spell1Id": 4,
                    "spell2Id": 32,
                    "stats": {
                        "win": False,
                        "kills": 1,
                        "deaths": 8,
                        "assists": 12,
                        "totalDamageDealtToChampions": 11000,
                        "totalDamageTaken": 18000,
                        "goldEarned": 9000,
                        "totalMinionsKilled": 25,
                    },
                },
                {
                    "participantId": 2,
                    "teamId": 100,
                    "championId": 99,
                    "spell1Id": 4,
                    "spell2Id": 32,
                    "stats": {
                        "win": True,
                        "kills": 9,
                        "deaths": 0,
                        "assists": 20,
                        "totalDamageDealtToChampions": 32000,
                        "totalDamageTaken": 9000,
                        "goldEarned": 13000,
                        "totalMinionsKilled": 55,
                        "item0": 6655,
                        "item1": 3020,
                    },
                },
            ],
            "participantIdentities": [
                {
                    "participantId": 1,
                    "player": {"currentAccountId": 111, "summonerName": "Other"},
                },
                {
                    "participantId": 2,
                    "player": {"currentAccountId": 222, "summonerName": "Me"},
                },
            ],
        }
        current = {"accountId": 222, "displayName": "Me"}

        row = normalize_match(game, current)

        self.assertIsNotNone(row)
        assert row is not None
        self.assertEqual("123", row["match_id"])
        self.assertEqual(99, row["champion_id"])
        self.assertTrue(row["win"])
        self.assertEqual(29.0, row["kda"])
        self.assertEqual(55, row["cs"])
        self.assertEqual("[6655, 3020]", row["items_json"])

    def test_stats_filter_and_champion_summary(self) -> None:
        rows = [
            {"queue_id": 450, "champion_name": "A", "win": True, "kda": 2.5},
            {"queue_id": 2400, "champion_name": "A", "win": False, "kda": 1.5},
            {"queue_id": 420, "champion_name": "B", "win": True, "kda": 4},
        ]

        aram_rows = filter_rows(rows)
        summary = build_summary(aram_rows)
        champs = champion_stats(aram_rows)

        self.assertEqual(2, summary["games"])
        self.assertEqual(0.5, summary["winrate"])
        self.assertEqual(1, len(champs))
        self.assertEqual("A", champs[0]["champion"])
        self.assertEqual(2.0, champs[0]["avg_kda"])

    def test_cn_aram_shape_is_detected(self) -> None:
        self.assertTrue(is_aram_game({"queueId": 2400, "gameMode": "KIWI", "mapId": 12}))
        self.assertTrue(is_aram_game({"queue_id": 2400, "game_mode": "KIWI"}))

    def test_normalize_participants_marks_sides(self) -> None:
        game = {
            "gameId": 321,
            "queueId": 2400,
            "gameMode": "KIWI",
            "mapId": 12,
            "participants": [
                {
                    "participantId": 1,
                    "teamId": 100,
                    "championId": 1,
                    "spell1Id": 4,
                    "spell2Id": 32,
                    "stats": {
                        "win": True,
                        "kills": 1,
                        "deaths": 1,
                        "assists": 1,
                        "champLevel": 18,
                        "goldSpent": 12000,
                        "item0": 6655,
                        "perk0": 8005,
                        "playerAugment1": 1,
                    },
                },
                {
                    "participantId": 2,
                    "teamId": 100,
                    "championId": 2,
                    "stats": {"win": True, "kills": 2, "deaths": 1, "assists": 1},
                },
                {
                    "participantId": 3,
                    "teamId": 200,
                    "championId": 3,
                    "stats": {"win": False, "kills": 3, "deaths": 1, "assists": 1},
                },
            ],
            "participantIdentities": [
                {
                    "participantId": 1,
                    "player": {"currentAccountId": 111, "gameName": "Me", "tagLine": "CN1"},
                },
                {
                    "participantId": 2,
                    "player": {"currentAccountId": 222, "gameName": "Ally", "tagLine": "CN1"},
                },
                {
                    "participantId": 3,
                    "player": {"currentAccountId": 333, "gameName": "Enemy", "tagLine": "CN1"},
                },
            ],
        }

        rows = normalize_match_participants(game, {"accountId": 111})
        sides = {row["riot_id"]: row["side"] for row in rows}

        self.assertEqual("self", sides["Me#CN1"])
        self.assertEqual("ally", sides["Ally#CN1"])
        self.assertEqual("enemy", sides["Enemy#CN1"])
        self.assertEqual(18, rows[0]["champ_level"])
        self.assertEqual(12000, rows[0]["gold_spent"])
        self.assertEqual(6655, rows[0]["item0"])
        self.assertEqual("[8005]", rows[0]["perks_json"])
        self.assertEqual("[1]", rows[0]["augments_json"])


if __name__ == "__main__":
    unittest.main()
