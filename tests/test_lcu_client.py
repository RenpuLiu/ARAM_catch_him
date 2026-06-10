from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from lcu_client import LCUClient, LCUError, _parse_process_args, parse_lockfile


class LCUClientTests(unittest.TestCase):
    def test_parse_lockfile_allows_colons_in_password(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "lockfile"
            path.write_text("LeagueClient:1234:56789:abc:def:https", encoding="utf-8")

            lock = parse_lockfile(path)

        self.assertEqual("LeagueClient", lock.name)
        self.assertEqual("1234", lock.pid)
        self.assertEqual("56789", lock.port)
        self.assertEqual("abc:def", lock.password)
        self.assertEqual("https", lock.protocol)

    def test_parse_lockfile_rejects_empty_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "lockfile"
            path.write_text("", encoding="utf-8")

            with self.assertRaisesRegex(LCUError, "empty"):
                parse_lockfile(path)

    def test_parse_process_args_keeps_token_padding(self) -> None:
        args = _parse_process_args(
            [
                "LeagueClientUx.exe",
                "--app-port=12345",
                "--remoting-auth-token=abc==",
                "--app-protocol=https",
            ]
        )

        self.assertEqual("12345", args["app-port"])
        self.assertEqual("abc==", args["remoting-auth-token"])
        self.assertEqual("https", args["app-protocol"])

    def test_matchlist_falls_back_to_v1_products_endpoint(self) -> None:
        class FakeClient:
            def __init__(self) -> None:
                self.paths: list[str] = []

            def get(self, path: str, **params):
                self.paths.append(path)
                if path == "/lol-match-history/v2/matchlist":
                    raise LCUError("404")
                return {"games": {"games": [{"gameId": 1}]}}

        fake = FakeClient()

        result = LCUClient.matchlist(fake, beg_index=0, end_index=1)

        self.assertEqual({"games": {"games": [{"gameId": 1}]}}, result)
        self.assertEqual(
            [
                "/lol-match-history/v2/matchlist",
                "/lol-match-history/v1/products/lol/current-summoner/matches",
            ],
            fake.paths,
        )


if __name__ == "__main__":
    unittest.main()
