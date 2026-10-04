"""Resolve release paths from the project directory, independently of cwd."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_project_path(path: str | Path) -> Path:
    """Keep absolute overrides; anchor relative paths at the release root."""
    path = Path(path).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path
