from __future__ import annotations

import base64
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class LCUError(RuntimeError):
    """Raised when the local League Client API cannot be reached."""


@dataclass(frozen=True)
class LockfileInfo:
    name: str
    pid: str
    port: str
    password: str
    protocol: str
    path: Path

    @property
    def base_url(self) -> str:
        return f"{self.protocol}://127.0.0.1:{self.port}"


def parse_lockfile(path: Path) -> LockfileInfo:
    raw = _read_lockfile_text(path)
    parts = raw.split(":")
    if len(parts) < 5:
        raise LCUError(
            f"Invalid lockfile format at {path}: expected at least 5 fields, got {len(parts)}"
        )

    name, pid, port = parts[:3]
    protocol = parts[-1]
    password = ":".join(parts[3:-1])
    _validate_lock(name, pid, port, password, protocol, str(path))
    return LockfileInfo(
        name=name,
        pid=pid,
        port=port,
        password=password,
        protocol=protocol,
        path=path,
    )


def _read_lockfile_text(path: Path) -> str:
    try:
        raw = path.read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise LCUError(f"Cannot read lockfile at {path}: {exc}") from exc

    text = raw.strip("\ufeff\x00\r\n \t")
    if not text:
        raise LCUError(f"Lockfile is empty at {path}")
    return text


def _validate_lock(
    name: str,
    pid: str,
    port: str,
    password: str,
    protocol: str,
    source: str,
) -> None:
    if not name:
        raise LCUError(f"Invalid lockfile format at {source}: missing name")
    if not pid.isdigit():
        raise LCUError(f"Invalid lockfile format at {source}: pid is not numeric")
    if not port.isdigit():
        raise LCUError(f"Invalid lockfile format at {source}: port is not numeric")
    if not password:
        raise LCUError(f"Invalid lockfile format at {source}: missing password")
    if protocol not in {"http", "https"}:
        raise LCUError(
            f"Invalid lockfile format at {source}: protocol must be http or https"
        )


def _candidate_paths_from_processes() -> list[Path]:
    try:
        import psutil
    except ImportError:
        return []

    candidates: list[Path] = []
    process_names = (
        "leagueclient",
        "leagueclientux",
        "league of legends",
    )

    for proc in psutil.process_iter(["name", "exe", "cwd"]):
        try:
            name = (proc.info.get("name") or "").lower()
            if not any(needle in name for needle in process_names):
                continue

            for raw_path in (proc.info.get("cwd"), proc.info.get("exe")):
                if not raw_path:
                    continue

                path = Path(raw_path)
                if path.suffix:
                    path = path.parent

                candidates.extend(
                    [
                        path / "lockfile",
                        path.parent / "lockfile",
                        path.parent / "LeagueClient" / "lockfile",
                        path.parent / "League of Legends" / "lockfile",
                    ]
                )
        except (psutil.Error, OSError, PermissionError):
            continue

    return candidates


def _common_lockfile_paths() -> list[Path]:
    roots = [
        Path(r"C:\Riot Games\League of Legends"),
        Path(r"C:\Program Files\Riot Games\League of Legends"),
        Path(r"C:\Program Files (x86)\Riot Games\League of Legends"),
        Path(r"C:\Program Files\lol-cn"),
        Path(r"C:\Program Files (x86)\lol-cn"),
        Path(r"C:\Program Files\lol-cn\LeagueClient"),
        Path(r"C:\Program Files (x86)\lol-cn\LeagueClient"),
        Path(r"C:\Program Files\Tencent Games\League of Legends"),
        Path(r"C:\Program Files (x86)\Tencent Games\League of Legends"),
        Path(r"C:\Program Files\Tencent\League of Legends"),
        Path(r"C:\Program Files (x86)\Tencent\League of Legends"),
    ]
    return [root / "lockfile" for root in roots]


def find_lockfile(manual_path: str | None = None) -> Path:
    return discover_lock(manual_path).path


def discover_lock(manual_path: str | None = None) -> LockfileInfo:
    if manual_path:
        path = Path(manual_path).expanduser()
        if not path.exists():
            raise LCUError(f"Lockfile not found: {path}")
        try:
            return parse_lockfile(path)
        except LCUError as exc:
            process_lock = _lock_from_process_args()
            if process_lock:
                return process_lock
            raise exc

    errors: list[str] = []
    seen: set[Path] = set()
    for candidate in [*_candidate_paths_from_processes(), *_common_lockfile_paths()]:
        path = candidate.expanduser()
        if path in seen:
            continue
        seen.add(path)
        if not path.exists():
            continue
        try:
            return parse_lockfile(path)
        except LCUError as exc:
            errors.append(str(exc))

    process_lock = _lock_from_process_args()
    if process_lock:
        return process_lock

    detail = f" Checked invalid candidates: {' | '.join(errors[:3])}" if errors else ""
    raise LCUError(
        "Cannot find League Client lockfile. Start the League client, log in, "
        "then pass --lockfile with the full lockfile path if needed."
        f"{detail}"
    )


def _lock_from_process_args() -> LockfileInfo | None:
    try:
        import psutil
    except ImportError:
        return None

    for proc in psutil.process_iter(["name", "pid", "cmdline", "exe", "cwd"]):
        try:
            name = (proc.info.get("name") or "").lower()
            if "leagueclient" not in name:
                continue

            args = _parse_process_args(proc.info.get("cmdline") or [])
            port = args.get("app-port")
            password = args.get("remoting-auth-token")
            protocol = args.get("app-protocol", "https")
            app_name = args.get("app-name", "LeagueClient")
            pid = args.get("app-pid") or str(proc.info.get("pid") or "")

            if not port or not password:
                continue

            _validate_lock(app_name, pid, port, password, protocol, f"process {pid}")
            return LockfileInfo(
                name=app_name,
                pid=pid,
                port=port,
                password=password,
                protocol=protocol,
                path=_process_lockfile_path(proc.info.get("cwd"), proc.info.get("exe")),
            )
        except (psutil.Error, OSError, PermissionError, LCUError):
            continue

    return None


def _parse_process_args(cmdline: list[str]) -> dict[str, str]:
    args: dict[str, str] = {}
    for raw_arg in cmdline:
        arg = raw_arg.strip()
        if not arg.startswith("--"):
            continue
        key, separator, value = arg[2:].partition("=")
        if separator:
            args[key] = value.strip('"')
    return args


def _process_lockfile_path(cwd: str | None, exe: str | None) -> Path:
    for raw_path in (cwd, exe):
        if not raw_path:
            continue
        path = Path(raw_path)
        if path.suffix:
            path = path.parent
        return path / "lockfile"
    return Path("process-args")


class LCUClient:
    def __init__(self, lockfile: LockfileInfo, timeout: float = 12.0):
        token = base64.b64encode(f"riot:{lockfile.password}".encode("utf-8")).decode(
            "ascii"
        )
        self.lockfile = lockfile
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Basic {token}",
                "Accept": "application/json",
                "User-Agent": "lol-aram-catch-him/0.1",
            }
        )

    def get(self, path: str, **params: Any) -> Any:
        url = f"{self.lockfile.base_url}{path}"
        try:
            response = self.session.get(
                url,
                params={k: v for k, v in params.items() if v is not None},
                verify=False,
                timeout=self.timeout,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise LCUError(f"LCU request failed: GET {path}: {exc}") from exc

        if not response.text:
            return None
        return response.json()

    def current_summoner(self) -> Any:
        return self.get("/lol-summoner/v1/current-summoner")

    def matchlist(self, beg_index: int = 0, end_index: int = 20) -> Any:
        endpoints = [
            "/lol-match-history/v2/matchlist",
            "/lol-match-history/v1/products/lol/current-summoner/matches",
        ]
        errors: list[str] = []
        for endpoint in endpoints:
            try:
                return self.get(endpoint, begIndex=beg_index, endIndex=end_index)
            except LCUError as exc:
                errors.append(str(exc))

        raise LCUError(f"All matchlist endpoints failed: {' | '.join(errors)}")

    def game_detail(self, game_id: int | str) -> Any:
        return self.get(f"/lol-match-history/v1/games/{game_id}")

    def game_timeline(self, game_id: int | str) -> Any:
        return self.get(f"/lol-match-history/v1/game-timelines/{game_id}")


def connect(lockfile_path: str | None = None) -> LCUClient:
    return LCUClient(discover_lock(lockfile_path))
