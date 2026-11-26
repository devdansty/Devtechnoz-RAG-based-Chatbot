"""Shared filesystem locations for the local project checkout."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
CONVERSATION_DIR = DATA_DIR / "conversations"
WEB_DIR = PROJECT_ROOT / "web"
