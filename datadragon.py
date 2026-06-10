from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import requests


BASE_URL = "https://ddragon.leagueoflegends.com"


def load_static_maps(
    cache_dir: str | Path = "data/cache",
    language: str = "zh_CN",
    refresh: bool = False,
) -> dict[str, dict[int, str]]:
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / f"ddragon_{language}.json"

    if target.exists() and not refresh:
        return _read_maps(target)

    version = _latest_version()
    payload = {
        "version": version,
        "champions": _champions(version, language),
        "items": _items(version, language),
        "summoner_spells": _summoner_spells(version, language),
    }
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return _maps_from_payload(payload)


def _latest_version() -> str:
    response = requests.get(f"{BASE_URL}/api/versions.json", timeout=20)
    response.raise_for_status()
    versions = response.json()
    if not versions:
        raise RuntimeError("Data Dragon versions list is empty")
    return versions[0]


def _champions(version: str, language: str) -> dict[str, Any]:
    url = f"{BASE_URL}/cdn/{version}/data/{language}/champion.json"
    response = requests.get(url, timeout=20)
    response.raise_for_status()
    return response.json().get("data", {})


def _items(version: str, language: str) -> dict[str, Any]:
    url = f"{BASE_URL}/cdn/{version}/data/{language}/item.json"
    response = requests.get(url, timeout=20)
    response.raise_for_status()
    return response.json().get("data", {})


def _summoner_spells(version: str, language: str) -> dict[str, Any]:
    url = f"{BASE_URL}/cdn/{version}/data/{language}/summoner.json"
    response = requests.get(url, timeout=20)
    response.raise_for_status()
    return response.json().get("data", {})


def _read_maps(path: Path) -> dict[str, dict[int, str]]:
    return _maps_from_payload(json.loads(path.read_text(encoding="utf-8")))


def _maps_from_payload(payload: dict[str, Any]) -> dict[str, dict[int, str]]:
    champions = {
        int(value["key"]): value.get("name", key)
        for key, value in payload.get("champions", {}).items()
        if str(value.get("key", "")).isdigit()
    }
    items = {
        int(key): value.get("name", key)
        for key, value in payload.get("items", {}).items()
        if str(key).isdigit()
    }
    spells = {
        int(value["key"]): value.get("name", key)
        for key, value in payload.get("summoner_spells", {}).items()
        if str(value.get("key", "")).isdigit()
    }
    return {"champions": champions, "items": items, "summoner_spells": spells}
