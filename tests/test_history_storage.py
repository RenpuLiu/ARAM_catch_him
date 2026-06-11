from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from lol_cn_lcu_dump import dump_lcu_data
from storage import matches_csv_path, participants_csv_path, read_rows_csv


class HistoryStorageTests(unittest.TestCase):
    def test_dump_lcu_data_merges_history_and_deduplicates_matches(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp)

            with patch("lol_cn_lcu_dump.connect", return_value=_FakeClient([1, 2])):
                first = dump_lcu_data(n=2, out=data)

            with patch("lol_cn_lcu_dump.connect", return_value=_FakeClient([2, 3])):
                second = dump_lcu_data(n=2, out=data)

            matches = read_rows_csv(matches_csv_path(data))
            participants = read_rows_csv(participants_csv_path(data))
            archived_ids = [
                data.joinpath("raw", "game_details", f"{game_id}.json").exists()
                for game_id in (1, 2, 3)
            ]

            self.assertEqual(2, first["total_matches"])
            self.assertEqual(2, first["added_matches"])
            self.assertEqual(3, second["total_matches"])
            self.assertEqual(1, second["added_matches"])
            self.assertEqual(1, second["updated_matches"])
            self.assertEqual(["3", "2", "1"], [row["match_id"] for row in matches])
            self.assertEqual(6, len(participants))
            self.assertEqual([True, True, True], archived_ids)


class _FakeClient:
    def __init__(self, game_ids: list[int]) -> None:
        self.game_ids = game_ids
        self.lockfile = SimpleNamespace(path=Path("fake-lockfile"))

    def current_summoner(self) -> dict:
        return {"accountId": 111, "displayName": "Me"}

    def matchlist(self, beg_index: int, end_index: int) -> dict:
        games = [{"gameId": game_id} for game_id in self.game_ids[beg_index:end_index]]
        return {"games": {"games": games}}

    def game_detail(self, game_id: int) -> dict:
        return _game(int(game_id))


def _game(game_id: int) -> dict:
    return {
        "gameId": game_id,
        "queueId": 450,
        "gameMode": "ARAM",
        "gameCreation": 1710000000000 + game_id,
        "gameDuration": 1200,
        "participants": [
            {
                "participantId": 1,
                "teamId": 100,
                "championId": 22,
                "spell1Id": 4,
                "spell2Id": 32,
                "stats": {
                    "win": game_id % 2 == 0,
                    "kills": game_id,
                    "deaths": 1,
                    "assists": 10,
                    "totalDamageDealtToChampions": 10000 + game_id,
                    "totalDamageTaken": 9000,
                    "goldEarned": 12000,
                },
            },
            {
                "participantId": 2,
                "teamId": 200,
                "championId": 99,
                "spell1Id": 4,
                "spell2Id": 32,
                "stats": {
                    "win": game_id % 2 != 0,
                    "kills": 3,
                    "deaths": 2,
                    "assists": 5,
                    "totalDamageDealtToChampions": 8000,
                    "totalDamageTaken": 7000,
                    "goldEarned": 10000,
                },
            },
        ],
        "participantIdentities": [
            {
                "participantId": 1,
                "player": {"currentAccountId": 111, "summonerName": "Me"},
            },
            {
                "participantId": 2,
                "player": {"currentAccountId": 222, "summonerName": "Enemy"},
            },
        ],
    }


if __name__ == "__main__":
    unittest.main()
