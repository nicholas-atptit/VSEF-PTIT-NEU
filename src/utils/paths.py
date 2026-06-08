"""Repository path helpers."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def repo_path(*parts: str | Path) -> Path:
    return REPO_ROOT.joinpath(*parts)


def ensure_directory(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def relative_to_repo(path: str | Path) -> str:
    target = Path(path)
    try:
        return target.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return target.as_posix()
