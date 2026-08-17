"""
storage.py — SQLite persistence for Persona
============================================
Persists characters (with emotion state, portrait, voice) and per-character
conversation transcripts to a local SQLite file (`persona.db`) so everything
survives a server restart. ChromaDB already persists semantic memory on its
own; this module covers the structured/relational state that previously lived
only in memory.

All functions are safe to call from FastAPI's threaded request handlers: a
single shared connection is guarded by a module-level lock.
"""

import os
import json
import sqlite3
import threading
from datetime import datetime, timezone

DB_PATH = os.getenv("PERSONA_DB", "persona.db")

_conn = sqlite3.connect(DB_PATH, check_same_thread=False)
_conn.row_factory = sqlite3.Row
_lock = threading.Lock()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def init_db() -> None:
    """Create tables if they don't exist. Idempotent."""
    with _lock:
        _conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS characters (
                name             TEXT PRIMARY KEY,
                description      TEXT NOT NULL DEFAULT '',
                personality_tags TEXT NOT NULL DEFAULT '[]',
                visual_style     TEXT NOT NULL DEFAULT '',
                mood             TEXT NOT NULL DEFAULT 'neutral',
                affinity         REAL NOT NULL DEFAULT 0.5,
                mood_intensity   REAL NOT NULL DEFAULT 0.5,
                image_dir        TEXT NOT NULL DEFAULT '',
                portrait_url     TEXT,
                voice            TEXT,
                created_at       TEXT NOT NULL DEFAULT ''
            );

            CREATE TABLE IF NOT EXISTS messages (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                character_name TEXT NOT NULL,
                role           TEXT NOT NULL,
                content        TEXT NOT NULL DEFAULT '',
                image_url      TEXT,
                kind           TEXT NOT NULL DEFAULT 'solo',
                created_at     TEXT NOT NULL DEFAULT ''
            );

            CREATE INDEX IF NOT EXISTS idx_messages_char
                ON messages(character_name, id);
            """
        )
        _conn.commit()


# --------------------------------------------------------------------------- #
# Characters                                                                  #
# --------------------------------------------------------------------------- #
def upsert_character(
    name: str,
    description: str,
    personality_tags: list[str],
    visual_style: str,
    image_dir: str,
    mood: str = "neutral",
    affinity: float = 0.5,
    mood_intensity: float = 0.5,
    portrait_url: str | None = None,
    voice: str | None = None,
) -> None:
    with _lock:
        _conn.execute(
            """
            INSERT INTO characters
                (name, description, personality_tags, visual_style, mood,
                 affinity, mood_intensity, image_dir, portrait_url, voice, created_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(name) DO UPDATE SET
                description=excluded.description,
                personality_tags=excluded.personality_tags,
                visual_style=excluded.visual_style,
                mood=excluded.mood,
                affinity=excluded.affinity,
                mood_intensity=excluded.mood_intensity,
                image_dir=excluded.image_dir,
                portrait_url=excluded.portrait_url,
                voice=excluded.voice
            """,
            (
                name, description, json.dumps(personality_tags), visual_style,
                mood, affinity, mood_intensity, image_dir, portrait_url, voice, _now(),
            ),
        )
        _conn.commit()


def update_fields(name: str, **fields) -> None:
    """Update arbitrary character columns (description, personality_tags,
    visual_style, voice, portrait_url, ...). personality_tags is JSON-encoded."""
    if not fields:
        return
    if "personality_tags" in fields and isinstance(fields["personality_tags"], list):
        fields["personality_tags"] = json.dumps(fields["personality_tags"])
    cols = ", ".join(f"{k}=?" for k in fields)
    with _lock:
        _conn.execute(f"UPDATE characters SET {cols} WHERE name=?",
                      (*fields.values(), name))
        _conn.commit()


def update_emotion(name: str, mood: str, affinity: float, mood_intensity: float) -> None:
    with _lock:
        _conn.execute(
            "UPDATE characters SET mood=?, affinity=?, mood_intensity=? WHERE name=?",
            (mood, affinity, mood_intensity, name),
        )
        _conn.commit()


def set_portrait(name: str, url: str) -> None:
    with _lock:
        _conn.execute("UPDATE characters SET portrait_url=? WHERE name=?", (url, name))
        _conn.commit()


def load_characters() -> list[dict]:
    """Return all characters as plain dicts (personality_tags decoded to a list)."""
    with _lock:
        rows = _conn.execute("SELECT * FROM characters ORDER BY created_at").fetchall()
    out = []
    for r in rows:
        d = dict(r)
        try:
            d["personality_tags"] = json.loads(d.get("personality_tags") or "[]")
        except (json.JSONDecodeError, TypeError):
            d["personality_tags"] = []
        out.append(d)
    return out


def delete_character(name: str) -> None:
    with _lock:
        _conn.execute("DELETE FROM messages WHERE character_name=?", (name,))
        _conn.execute("DELETE FROM characters WHERE name=?", (name,))
        _conn.commit()


# --------------------------------------------------------------------------- #
# Messages                                                                    #
# --------------------------------------------------------------------------- #
def add_message(character_name: str, role: str, content: str,
                image_url: str | None = None, kind: str = "solo") -> None:
    with _lock:
        _conn.execute(
            """INSERT INTO messages (character_name, role, content, image_url, kind, created_at)
               VALUES (?,?,?,?,?,?)""",
            (character_name, role, content, image_url, kind, _now()),
        )
        _conn.commit()


def get_messages(character_name: str, kinds: tuple[str, ...] = ("solo",)) -> list[dict]:
    """Transcript rows for a character, restricted to the given `kind`s.

    Defaults to solo only: group-chat turns are shared chatter that would
    otherwise replay inside a one-on-one conversation.
    """
    placeholders = ",".join("?" * len(kinds))
    with _lock:
        rows = _conn.execute(
            f"SELECT role, content, image_url, kind, created_at "
            f"FROM messages WHERE character_name=? AND kind IN ({placeholders}) "
            f"ORDER BY id",
            (character_name, *kinds),
        ).fetchall()
    return [dict(r) for r in rows]


def clear_messages(character_name: str) -> None:
    with _lock:
        _conn.execute("DELETE FROM messages WHERE character_name=?", (character_name,))
        _conn.commit()


def reset_all() -> None:
    """Reset Session: drop all transcripts and reset every character's emotion
    to neutral/0.5, but keep the characters themselves."""
    with _lock:
        _conn.execute("DELETE FROM messages")
        _conn.execute(
            "UPDATE characters SET mood='neutral', affinity=0.5, mood_intensity=0.5"
        )
        _conn.commit()
