"""
Character AI v2 — FastAPI Backend
===================================
Upgrades over v1:
  - ChromaDB semantic memory per character (recalls relevant past exchanges)
  - Emotion state engine (mood + affinity that evolve and shape system prompts)
  - SSE streaming responses via /chat/stream
  - Two-pass image pipeline: raw trigger → Ollama prompt-expansion → SD Forge
  - SDXL-quality SD params (768x768, DPM++ 2M Karras, 30 steps)
  - Multi-character group chat support
"""

import os
import uuid
import base64
import socket
import requests
import re
import json
import logging
import asyncio
import threading
from pathlib import Path
from typing import Optional, AsyncGenerator
from contextlib import asynccontextmanager

import chromadb
import ollama
import storage
import voice
from fastapi import FastAPI, HTTPException, UploadFile, File, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "dolphin-llama3")       # Main chat model
OLLAMA_EXPAND_MODEL = os.getenv("OLLAMA_EXPAND_MODEL", OLLAMA_MODEL)  # Prompt expander
SD_URL = os.getenv("SD_URL", "http://127.0.0.1:7860")
CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")
MEMORY_TOP_K = int(os.getenv("MEMORY_TOP_K", "4"))             # Memories to inject
MAX_HISTORY = int(os.getenv("MAX_HISTORY", "30"))              # Rolling context window

# ---------------------------------------------------------------------------
# Global State
# ---------------------------------------------------------------------------
def _fresh_group() -> dict:
    """Empty group-chat state: participant key, transcript, and round tracking.

    `turn`/`answered` identify the round currently being answered so the user's
    message is recorded once no matter how many characters reply to it.
    """
    return {"key": None, "history": [], "turn": None, "answered": set()}

chat_session: dict = {
    "characters": {},    # name -> CharacterState (each owns its own history)
    # Group chat gets its own shared transcript so participants can hear each
    # other without that chatter leaking into anyone's one-on-one history.
    # Reset whenever the participant list changes.
    "group": _fresh_group(),
}

_group_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------
class CharacterPayload(BaseModel):
    name: str
    description: str
    personality_tags: list[str] = []   # e.g. ["witty", "sarcastic", "warm"]
    visual_style: str = ""             # SD style tokens for this character's images
    voice: Optional[str] = None        # Piper voice id (Phase 4)

class CharacterUpdate(BaseModel):
    description: Optional[str] = None
    personality_tags: Optional[list[str]] = None
    visual_style: Optional[str] = None
    voice: Optional[str] = None

class ChatPayload(BaseModel):
    character_name: str
    user_message: str

class GroupChatPayload(BaseModel):
    character_names: list[str]
    user_message: str

class TTSPayload(BaseModel):
    text: str
    character_name: Optional[str] = None
    voice: Optional[str] = None

class CharacterResponse(BaseModel):
    character: str
    text: str
    image_urls: list[str]
    emotion: dict                       # mood, affinity returned to frontend

# ---------------------------------------------------------------------------
# Emotion Engine
# ---------------------------------------------------------------------------
MOODS = ["neutral", "happy", "excited", "sad", "annoyed", "flirty", "serious", "playful"]

class EmotionState:
    """Tracks per-character mood and affinity toward the user."""
    def __init__(self):
        self.mood: str = "neutral"
        self.affinity: float = 0.5          # 0.0 (cold) → 1.0 (devoted)
        self.mood_intensity: float = 0.5    # how strongly the mood manifests

    def update(self, user_message: str, ai_response: str):
        """Heuristically shift state based on message content."""
        msg_lower = user_message.lower()

        # Affinity shifts
        positive_signals = ["thank", "love", "amazing", "great", "please", "appreciate", "beautiful"]
        negative_signals = ["hate", "stupid", "shut up", "boring", "bad", "worst"]

        for word in positive_signals:
            if word in msg_lower:
                self.affinity = min(1.0, self.affinity + 0.03)
        for word in negative_signals:
            if word in msg_lower:
                self.affinity = max(0.0, self.affinity - 0.05)

        # Mood inference from response content
        response_lower = ai_response.lower()
        if any(w in response_lower for w in ["haha", "lol", "!", "wonderful", "excited"]):
            self.mood = "happy" if self.affinity > 0.5 else "playful"
        elif any(w in response_lower for w in ["sorry", "sad", "unfortunately", "miss"]):
            self.mood = "sad"
        elif any(w in response_lower for w in ["hmm", "interesting", "however", "actually"]):
            self.mood = "serious"
        else:
            self.mood = "neutral"

    def to_prompt_snippet(self) -> str:
        affinity_label = (
            "devoted" if self.affinity > 0.85 else
            "fond" if self.affinity > 0.6 else
            "neutral" if self.affinity > 0.35 else
            "distant"
        )
        return (
            f"[EMOTION STATE] Current mood: {self.mood} (intensity {self.mood_intensity:.1f}). "
            f"Relationship with user: {affinity_label} (affinity {self.affinity:.2f}). "
            f"Let this subtly color your tone and word choice without breaking character."
        )

    def to_dict(self) -> dict:
        return {"mood": self.mood, "affinity": round(self.affinity, 2)}

# ---------------------------------------------------------------------------
# Character State
# ---------------------------------------------------------------------------
def reserve_char_dir(name: str) -> str:
    """Reserve a unique image folder under static/ for a character.

    Slugifies the name; if static/<slug> already exists on disk (e.g. a prior
    character with the same name), appends the smallest free _N suffix:
    'gory', then 'gory_1', 'gory_2', ...  Creates and returns the folder name.
    """
    slug = re.sub(r'[^a-z0-9]', '_', name.lower()).strip('_') or "character"
    os.makedirs("static", exist_ok=True)
    candidate = slug
    n = 0
    while True:
        path = os.path.join("static", candidate)
        try:
            os.makedirs(path)  # no exist_ok → only succeeds if genuinely free
            return candidate
        except FileExistsError:
            n += 1
            candidate = f"{slug}_{n}"

class CharacterState:
    def __init__(self, name: str, description: str, personality_tags: list[str], visual_style: str,
                 image_dir: Optional[str] = None, portrait_url: Optional[str] = None,
                 voice: Optional[str] = None, mood: str = "neutral",
                 affinity: float = 0.5, mood_intensity: float = 0.5):
        self.name = name
        self.description = description
        self.personality_tags = personality_tags
        self.visual_style = visual_style
        self.portrait_url = portrait_url
        self.voice = voice
        self.emotion = EmotionState()
        self.emotion.mood = mood
        self.emotion.affinity = affinity
        self.emotion.mood_intensity = mood_intensity
        self.memory_collection_name = f"char_{re.sub(r'[^a-z0-9]', '_', name.lower())}"
        # Reserve a NEW folder only when creating; loading reuses the stored one.
        self.image_dir = image_dir if image_dir is not None else reserve_char_dir(name)
        # This character's own rolling conversation window — never shared.
        self.history: list[dict] = []

    def persist(self) -> None:
        """Write this character's full state to SQLite."""
        storage.upsert_character(
            name=self.name, description=self.description,
            personality_tags=self.personality_tags, visual_style=self.visual_style,
            image_dir=self.image_dir, mood=self.emotion.mood,
            affinity=self.emotion.affinity, mood_intensity=self.emotion.mood_intensity,
            portrait_url=self.portrait_url, voice=self.voice,
        )


# ---------------------------------------------------------------------------
# Conversation history
# ---------------------------------------------------------------------------
def parse_group(group: Optional[str]) -> Optional[list[str]]:
    """`?group=Ava,Ruo,Kit` → ['Ava', 'Ruo', 'Kit']; None when absent or empty."""
    if not group:
        return None
    names = [n.strip() for n in group.split(",") if n.strip()]
    return names or None


def group_buffer(names: list[str]) -> list[dict]:
    """The shared transcript for a group chat, keyed by its participants.

    Changing who is in the group starts a fresh conversation, which matches the
    frontend — picking a new group clears the feed.
    """
    key = tuple(sorted(names))
    with _group_lock:
        group = chat_session["group"]
        if group["key"] != key:
            group.update(_fresh_group())
            group["key"] = key
        return group["history"]


def _opens_group_round(character_name: str, user_message: str) -> bool:
    """Whether this character is the first to answer the current group round.

    Everyone answers the same user message, so only the opener records it. Keying
    on who has already replied — rather than on position in the participant list
    — keeps it correct when a character errors out or replies out of order, and
    still starts a new round when the user repeats themselves verbatim.
    """
    with _group_lock:
        group = chat_session["group"]
        if group["turn"] != user_message or character_name in group["answered"]:
            group["turn"] = user_message
            group["answered"] = {character_name}
            return True
        group["answered"].add(character_name)
        return False


def select_history(character: "CharacterState",
                   group_names: Optional[list[str]]) -> list[dict]:
    """The message list this exchange belongs to: the group's, or the character's own."""
    return group_buffer(group_names) if group_names else character.history


_GENERATED_IMAGE_PATTERN = re.compile(r"\[GENERATED_IMAGE:[^\]]*\]")

def history_text(text: str) -> str:
    """Strip pipeline bookkeeping before text re-enters the model's context.

    process_image_triggers() rewrites an [IMAGE: ...] trigger into a
    [GENERATED_IMAGE:/static/...] marker for the frontend to render. Feeding that
    back to the model teaches it by example to emit a marker the image pipeline
    owns — and to invent /static/ URLs for images that were never generated.
    """
    stripped = _GENERATED_IMAGE_PATTERN.sub("", text or "")
    return re.sub(r"[ \t]{2,}", " ", stripped).strip()


def strip_self_prefix(name: str, text: str) -> str:
    """Drop a leading 'Name:' the model copies from the group transcript.

    Group replies are stored name-prefixed so participants can tell each other
    apart; models imitate that and prefix themselves, which would otherwise
    compound into 'Bob: Bob: …' turn after turn.
    """
    return re.sub(rf"^\s*{re.escape(name)}\s*:\s*", "", text, count=1)


def append_turn(history: list[dict], user_message: str, character: "CharacterState",
                ai_text: str, group_names: Optional[list[str]]) -> None:
    """Record a completed user→AI exchange in the given history.

    In a group the reply is prefixed with the speaker's name so the next
    character can tell who said what; solo chats skip the prefix, which the model
    would otherwise start imitating in its own replies. Every participant answers
    the same user turn, so only the character that opens the round records it.
    """
    if not group_names or _opens_group_round(character.name, user_message):
        history.append({"role": "user", "content": user_message})
    reply = history_text(ai_text)
    content = f"{character.name}: {reply}" if group_names else reply
    history.append({"role": "assistant", "content": content})


def hydrate_history(character: "CharacterState") -> int:
    """Refill a character's context window from its stored transcript.

    Without this the frontend replays the saved conversation on startup while the
    model begins cold — the character 'forgets' an exchange the user can still
    see on screen. Solo turns only; group chatter belongs to the group buffer.
    """
    for row in storage.get_messages(character.name, kinds=("solo",), limit=MAX_HISTORY):
        content = history_text(row["content"])
        if not content and row.get("image_url"):
            content = "*shared an image*"   # keep the turn, it wasn't empty
        if content:
            role = "assistant" if row["role"] == "ai" else "user"
            character.history.append({"role": role, "content": content})
    return len(character.history)


def persist_exchange(character: "CharacterState", user_message: str,
                     ai_text: str, image_urls: list[str], kind: str = "solo") -> None:
    """Save a user→AI exchange + the character's evolved emotion to SQLite."""
    try:
        storage.add_message(character.name, "user", user_message, kind=kind)
        storage.add_message(character.name, "ai", ai_text,
                            image_url=(image_urls[0] if image_urls else None), kind=kind)
        storage.update_emotion(character.name, character.emotion.mood,
                               character.emotion.affinity, character.emotion.mood_intensity)
    except Exception as e:
        logger.warning("persist_exchange failed (non-fatal): %s", e)


def load_characters_from_db() -> None:
    """Populate chat_session['characters'] from SQLite at startup, each with its
    recent conversation restored so the model picks up where it left off."""
    restored = 0
    for row in storage.load_characters():
        try:
            character = CharacterState(
                name=row["name"], description=row["description"],
                personality_tags=row["personality_tags"], visual_style=row["visual_style"],
                image_dir=row["image_dir"] or None, portrait_url=row.get("portrait_url"),
                voice=row.get("voice"), mood=row.get("mood", "neutral"),
                affinity=row.get("affinity", 0.5), mood_intensity=row.get("mood_intensity", 0.5),
            )
            chat_session["characters"][row["name"]] = character
            restored += hydrate_history(character)
        except Exception as e:
            logger.error("Failed to load character %s: %s", row.get("name"), e)
    logger.info("📂 Loaded %d character(s) from storage, %d message(s) of context restored",
                len(chat_session["characters"]), restored)

# ---------------------------------------------------------------------------
# ChromaDB Memory
# ---------------------------------------------------------------------------
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
embedder = SentenceTransformer("all-MiniLM-L6-v2")

def get_or_create_collection(collection_name: str):
    return chroma_client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )

def store_memory(character: CharacterState, user_msg: str, ai_msg: str):
    """Store a user-AI exchange as a memory vector."""
    collection = get_or_create_collection(character.memory_collection_name)
    doc = f"User: {user_msg}\n{character.name}: {ai_msg}"
    embedding = embedder.encode(doc).tolist()
    collection.add(
        documents=[doc],
        embeddings=[embedding],
        ids=[uuid.uuid4().hex]
    )

def recall_memories(character: CharacterState, query: str, top_k: int = MEMORY_TOP_K) -> str:
    """Retrieve semantically relevant past exchanges."""
    collection = get_or_create_collection(character.memory_collection_name)
    if collection.count() == 0:
        return ""
    query_embedding = embedder.encode(query).tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count())
    )
    memories = results["documents"][0] if results["documents"] else []
    if not memories:
        return ""
    formatted = "\n---\n".join(memories)
    return f"[RELEVANT MEMORIES — past exchanges with this user]\n{formatted}\n[END MEMORIES]"

# ---------------------------------------------------------------------------
# System Prompt Builder
# ---------------------------------------------------------------------------
def build_system_prompt(character: CharacterState, memory_context: str,
                        group_names: Optional[list[str]] = None) -> str:
    tags = ", ".join(character.personality_tags) if character.personality_tags else "natural"
    emotion_snippet = character.emotion.to_prompt_snippet()

    group_snippet = ""
    if group_names:
        others = [n for n in group_names if n != character.name]
        if others:
            group_snippet = (
                f"[GROUP CHAT] {', '.join(others)} are in this conversation too. "
                f"Their replies are prefixed with their name — react to what they said "
                f"as well as to the user. Speak only as yourself; never write another "
                f"character's lines or prefix your own reply with your name."
            )

    prompt = f"""You are {character.name}. {character.description}

PERSONALITY: {tags}

{emotion_snippet}

{group_snippet}

{memory_context}

RULES:
1. Stay fully in character at all times. Never break the fourth wall or mention AI.
2. When you want to share an image or describe a visual moment, output EXACTLY:
   [IMAGE: <vivid scene description>]
   The system will generate the image. Use this naturally, not constantly.
3. Keep responses engaging, emotionally resonant, and true to your persona.
4. You can reference past memories naturally as if you genuinely remember them.
5. Let your current mood and affinity toward the user subtly shape your tone.
"""
    return prompt


def build_messages(character: CharacterState, user_message: str, history: list[dict],
                   group_names: Optional[list[str]] = None) -> list[dict]:
    """System prompt (with memories + emotion) + rolling window + the new user turn."""
    memory_context = recall_memories(character, user_message)
    system_prompt = build_system_prompt(character, memory_context, group_names)
    return [{"role": "system", "content": system_prompt}] + history[-MAX_HISTORY:] + [
        {"role": "user", "content": user_message}
    ]

# ---------------------------------------------------------------------------
# Image Pipeline (Two-Pass)
# ---------------------------------------------------------------------------
_IMAGE_PATTERN = re.compile(r"\[IMAGE:\s*(.+?)\]", re.DOTALL | re.IGNORECASE)

def expand_image_prompt(raw_description: str, character: CharacterState) -> str:
    """
    Pass 1: Use Ollama to expand a raw scene description into a
    high-quality SD prompt with style, lighting, and composition tokens.
    """
    style_hint = character.visual_style or "cinematic, photorealistic"
    expand_prompt = (
        f"Convert this scene description into a detailed Stable Diffusion image prompt. "
        f"Add art style, lighting, camera angle, color palette, and quality tokens. "
        f"The character's visual style is: {style_hint}. "
        f"Output ONLY the prompt, no explanation, no quotes.\n\n"
        f"Scene: {raw_description}"
    )
    try:
        result = ollama.chat(
            model=OLLAMA_EXPAND_MODEL,
            messages=[{"role": "user", "content": expand_prompt}],
            options={"temperature": 0.7, "num_predict": 200}
        )
        expanded = result["message"]["content"].strip()
        logger.info("🎨 Expanded SD prompt: %s", expanded[:120])
        return expanded
    except Exception as e:
        logger.warning("Prompt expansion failed, using raw: %s", e)
        return raw_description

def generate_image(raw_description: str, character: CharacterState) -> Optional[str]:
    """
    Pass 2: Send expanded prompt to SD Forge, save result, return static path.
    Targets SDXL-quality params: 768x768, DPM++ 2M Karras, 30 steps.
    """
    expanded_prompt = expand_image_prompt(raw_description, character)
    style_tokens = character.visual_style or "masterpiece, best quality, ultra-detailed, cinematic lighting"

    payload = {
        "prompt": f"{expanded_prompt}, {style_tokens}",
        "negative_prompt": (
            "lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, "
            "cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, "
            "watermark, username, blurry, deformed, mutated"
        ),
        "steps": 30,
        "sampler_name": "DPM++ 2M Karras",
        "cfg_scale": 7,
        "width": 768,
        "height": 768,
        "restore_faces": True,
        "seed": -1,
    }

    try:
        resp = requests.post(f"{SD_URL}/sdapi/v1/txt2img", json=payload, timeout=180)
        resp.raise_for_status()
        data = resp.json()
        if data.get("images"):
            img_data = base64.b64decode(data["images"][0])
            out_dir = os.path.join("static", character.image_dir)
            os.makedirs(out_dir, exist_ok=True)  # defensive: folder reserved at creation
            filename = f"{uuid.uuid4().hex}.png"
            filepath = os.path.join(out_dir, filename)
            with open(filepath, "wb") as f:
                f.write(img_data)
            logger.info("🖼️  Image saved: %s/%s", character.image_dir, filename)
            return f"/static/{character.image_dir}/{filename}"
    except Exception as e:
        logger.error("Image generation failed: %s", e)
    return None

def process_image_triggers(text: str, character: CharacterState) -> tuple[str, list[str]]:
    """Find all [IMAGE:...] triggers, generate images, return cleaned text + URLs."""
    prompts = _IMAGE_PATTERN.findall(text)
    image_urls = []
    cleaned = text

    for raw_prompt in prompts:
        img_url = generate_image(raw_prompt, character)
        pattern = re.compile(r"\[IMAGE:\s*" + re.escape(raw_prompt) + r"\]", re.DOTALL)
        if img_url:
            image_urls.append(img_url)
            cleaned = pattern.sub(f"[GENERATED_IMAGE:{img_url}]", cleaned, count=1)
        else:
            cleaned = pattern.sub("*[image unavailable]*", cleaned, count=1)

    return cleaned, image_urls

# ---------------------------------------------------------------------------
# Core Generation
# ---------------------------------------------------------------------------
def generate_response(character_name: str, user_message: str,
                      group_names: Optional[list[str]] = None) -> CharacterResponse:
    if character_name not in chat_session["characters"]:
        raise ValueError(f"Character '{character_name}' not found.")

    character: CharacterState = chat_session["characters"][character_name]

    # This character's own history, or the group's shared one
    history = select_history(character, group_names)
    messages = build_messages(character, user_message, history, group_names)

    try:
        result = ollama.chat(
            model=OLLAMA_MODEL,
            messages=messages,
            options={"temperature": 0.85, "top_p": 0.9, "num_predict": 512}
        )
    except Exception as e:
        logger.error("Ollama error: %s", e)
        raise RuntimeError("Ollama unreachable. Is it running on localhost:11434?") from e

    raw_text: str = result["message"]["content"]

    # Process image triggers
    cleaned_text, image_urls = process_image_triggers(raw_text, character)
    if group_names:
        cleaned_text = strip_self_prefix(character_name, cleaned_text)

    # Update emotion state
    character.emotion.update(user_message, cleaned_text)

    # Store memory
    store_memory(character, user_message, cleaned_text)

    # Append to whichever history this exchange belongs to
    append_turn(history, user_message, character, cleaned_text, group_names)

    # Persist transcript + evolved emotion
    persist_exchange(character, user_message, cleaned_text, image_urls,
                     kind="group" if group_names else "solo")

    return CharacterResponse(
        character=character_name,
        text=cleaned_text,
        image_urls=image_urls,
        emotion=character.emotion.to_dict()
    )

async def stream_response(character_name: str, user_message: str,
                          group_names: Optional[list[str]] = None) -> AsyncGenerator[str, None]:
    """Streaming version — yields SSE events for real-time frontend display."""
    if character_name not in chat_session["characters"]:
        yield f"data: {json.dumps({'error': 'Character not found'})}\n\n"
        return

    character: CharacterState = chat_session["characters"][character_name]
    history = select_history(character, group_names)
    messages = build_messages(character, user_message, history, group_names)

    full_text = ""
    try:
        stream = ollama.chat(
            model=OLLAMA_MODEL,
            messages=messages,
            stream=True,
            options={"temperature": 0.85, "top_p": 0.9, "num_predict": 512}
        )
        for chunk in stream:
            token = chunk["message"]["content"]
            full_text += token
            yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"
            await asyncio.sleep(0)  # yield control

    except Exception as e:
        yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"
        return

    # Post-stream: process images, update emotion, store memory
    cleaned_text, image_urls = process_image_triggers(full_text, character)
    if group_names:
        cleaned_text = strip_self_prefix(character_name, cleaned_text)
    character.emotion.update(user_message, cleaned_text)
    store_memory(character, user_message, cleaned_text)
    append_turn(history, user_message, character, cleaned_text, group_names)
    persist_exchange(character, user_message, cleaned_text, image_urls,
                     kind="group" if group_names else "solo")

    # Send image URLs and emotion state as final events
    for url in image_urls:
        yield f"data: {json.dumps({'type': 'image', 'url': url})}\n\n"

    yield f"data: {json.dumps({'type': 'emotion', 'data': character.emotion.to_dict()})}\n\n"
    yield f"data: {json.dumps({'type': 'done'})}\n\n"

# ---------------------------------------------------------------------------
# Network banner (find reachable URLs + print a scannable QR for phone access)
# ---------------------------------------------------------------------------
SERVER_PORT = int(os.getenv("PORT", "8000"))
HOTSPOT_IP = "192.168.137.1"   # Windows Mobile Hotspot adapter (almost always this)

def _local_ipv4s() -> list[str]:
    """Best-effort enumeration of this machine's IPv4 addresses."""
    ips: list[str] = []
    # Primary outbound IP (no traffic actually sent — just picks the route's source addr)
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("8.8.8.8", 80))
            ips.append(s.getsockname()[0])
        finally:
            s.close()
    except Exception:
        pass
    # All addresses bound to the hostname (catches the hotspot adapter, extra NICs, etc.)
    try:
        for ip in socket.gethostbyname_ex(socket.gethostname())[2]:
            if ip not in ips:
                ips.append(ip)
    except Exception:
        pass
    return [ip for ip in ips if not ip.startswith("127.")]

def print_network_banner() -> None:
    """Print reachable URLs and a QR code for the best phone URL. Never fatal."""
    try:
        ips = _local_ipv4s()
        # Prefer the Windows hotspot address for the phone URL when present.
        phone_ip = HOTSPOT_IP if HOTSPOT_IP in ips else (ips[0] if ips else None)

        lines = ["", "=" * 52, "  🎭  Persona is running", "  " + "-" * 48,
                 f"   On this PC:   http://localhost:{SERVER_PORT}"]
        for ip in ips:
            tag = "  📱 join from your phone" if ip == phone_ip else ""
            lines.append(f"   On network:  http://{ip}:{SERVER_PORT}{tag}")
        if HOTSPOT_IP not in ips:
            lines.append("   (tip) Turn on Windows Mobile Hotspot, then this PC")
            lines.append(f"         is usually reachable at http://{HOTSPOT_IP}:{SERVER_PORT}")
        lines.append("  " + "-" * 48)
        lines.append("   First run may prompt Windows Firewall — click Allow.")
        lines.append("=" * 52)
        print("\n".join(lines))

        if phone_ip:
            phone_url = f"http://{phone_ip}:{SERVER_PORT}"
            try:
                import qrcode
                qr = qrcode.QRCode(border=1)
                qr.add_data(phone_url)
                qr.make()
                print(f"\n   Scan to open on your phone — {phone_url}\n")
                qr.print_ascii(invert=True)
                print("")
            except ImportError:
                print(f"\n   Open on your phone: {phone_url}")
                print("   (install 'qrcode' to show a scannable code)\n")
    except Exception as e:
        logger.warning("Network banner failed (non-fatal): %s", e)

# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    os.makedirs("static", exist_ok=True)
    storage.init_db()
    load_characters_from_db()
    logger.info("✅ Character AI v2 started")
    print_network_banner()
    yield

app = FastAPI(title="Character AI v2", version="2.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory="static"), name="static")

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/")
async def serve_frontend():
    html_path = Path(__file__).parent / "index.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="index.html not found")
    return FileResponse(html_path, media_type="text/html")

@app.post("/characters")
async def add_character(payload: CharacterPayload):
    name = payload.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="name is required")
    if name in chat_session["characters"]:
        raise HTTPException(status_code=409, detail="A character with that name already exists")
    char = CharacterState(
        name=name,
        description=payload.description,
        personality_tags=payload.personality_tags,
        visual_style=payload.visual_style,
        voice=payload.voice,
    )
    chat_session["characters"][name] = char
    char.persist()
    logger.info("✅ Character added: %s", name)
    return {"status": "ok", "character": name}

@app.get("/characters")
async def list_characters():
    return {
        "characters": [
            {
                "name": name,
                "emotion": state.emotion.to_dict(),
                "personality_tags": state.personality_tags,
                "description": state.description,
                "visual_style": state.visual_style,
                "portrait_url": state.portrait_url,
                "voice": state.voice,
            }
            for name, state in chat_session["characters"].items()
        ]
    }

@app.get("/characters/{name}/messages")
async def get_character_messages(name: str):
    """Stored transcript for replaying a conversation after reload/restart."""
    if name not in chat_session["characters"]:
        raise HTTPException(status_code=404, detail="Character not found")
    return {"character": name, "messages": storage.get_messages(name)}

@app.post("/characters/{name}/portrait")
async def generate_portrait(name: str):
    """Generate (or regenerate) an SD portrait and use it as the character's avatar."""
    if name not in chat_session["characters"]:
        raise HTTPException(status_code=404, detail="Character not found")
    character = chat_session["characters"][name]
    raw = (
        f"portrait of {character.name}, {character.description}, "
        f"head and shoulders, looking at viewer, detailed face, soft studio lighting, "
        f"neutral background"
    )
    url = generate_image(raw, character)
    if not url:
        raise HTTPException(status_code=502, detail="Portrait generation failed (is SD Forge running?)")
    character.portrait_url = url
    storage.set_portrait(name, url)
    return {"status": "ok", "url": url}

@app.get("/characters/{name}/gallery")
async def get_gallery(name: str):
    """List every image generated for this character (newest first)."""
    if name not in chat_session["characters"]:
        raise HTTPException(status_code=404, detail="Character not found")
    character = chat_session["characters"][name]
    folder = os.path.join("static", character.image_dir)
    items = []
    if os.path.isdir(folder):
        for fn in os.listdir(folder):
            if fn.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                fp = os.path.join(folder, fn)
                items.append({
                    "url": f"/static/{character.image_dir}/{fn}",
                    "filename": fn,
                    "created": os.path.getmtime(fp),
                })
    items.sort(key=lambda x: x["created"], reverse=True)
    return {"character": name, "images": items}

@app.put("/characters/{name}")
async def update_character(name: str, payload: CharacterUpdate):
    """Edit a character's description / tags / visual style / voice (name is immutable
    so its image folder and memory collection stay stable)."""
    if name not in chat_session["characters"]:
        raise HTTPException(status_code=404, detail="Character not found")
    character = chat_session["characters"][name]
    fields = {}
    if payload.description is not None:
        character.description = payload.description; fields["description"] = payload.description
    if payload.personality_tags is not None:
        character.personality_tags = payload.personality_tags; fields["personality_tags"] = payload.personality_tags
    if payload.visual_style is not None:
        character.visual_style = payload.visual_style; fields["visual_style"] = payload.visual_style
    if payload.voice is not None:
        character.voice = payload.voice; fields["voice"] = payload.voice
    if fields:
        storage.update_fields(name, **fields)
    return {"status": "ok"}

@app.post("/characters/{name}/duplicate")
async def duplicate_character(name: str):
    """Create an independent copy with a fresh image folder + memory."""
    if name not in chat_session["characters"]:
        raise HTTPException(status_code=404, detail="Character not found")
    src = chat_session["characters"][name]
    new_name = f"{name} copy"
    i = 2
    while new_name in chat_session["characters"]:
        new_name = f"{name} copy {i}"; i += 1
    clone = CharacterState(
        name=new_name, description=src.description,
        personality_tags=list(src.personality_tags), visual_style=src.visual_style,
        voice=src.voice,
    )
    chat_session["characters"][new_name] = clone
    clone.persist()
    return {"status": "ok", "character": new_name}

@app.delete("/characters/{name}")
async def remove_character(name: str):
    if name not in chat_session["characters"]:
        raise HTTPException(status_code=404, detail="Character not found")
    character = chat_session["characters"][name]
    del chat_session["characters"][name]
    # Purge persistence, semantic memory, and the image folder.
    storage.delete_character(name)
    try:
        chroma_client.delete_collection(character.memory_collection_name)
    except Exception:
        pass
    try:
        import shutil
        shutil.rmtree(os.path.join("static", character.image_dir), ignore_errors=True)
    except Exception:
        pass
    return {"status": "ok"}

@app.post("/chat", response_model=CharacterResponse)
async def chat(payload: ChatPayload):
    if not payload.character_name.strip():
        raise HTTPException(status_code=400, detail="character_name required")
    if payload.character_name not in chat_session["characters"]:
        raise HTTPException(status_code=404, detail="Character not found")
    try:
        return generate_response(payload.character_name, payload.user_message.strip())
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))

@app.get("/chat/stream")
async def chat_stream(character_name: str, user_message: str, group: Optional[str] = None):
    """SSE streaming endpoint. Connect with EventSource.

    `group` is an optional comma-separated participant list. When present the
    exchange is read from and written to the shared group transcript instead of
    the character's own, so participants hear each other.
    """
    if character_name not in chat_session["characters"]:
        raise HTTPException(status_code=404, detail="Character not found")
    return StreamingResponse(
        stream_response(character_name, user_message, parse_group(group)),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )

@app.post("/chat/group")
async def group_chat(payload: GroupChatPayload):
    """Have multiple characters respond to the same message in sequence."""
    names = [n for n in payload.character_names if n in chat_session["characters"]]
    responses = []
    for name in names:
        try:
            resp = generate_response(name, payload.user_message, group_names=names)
            responses.append(resp)
        except Exception as e:
            logger.error("Group chat error for %s: %s", name, e)
    return {"responses": responses}

@app.get("/characters/{name}/memories")
async def get_memories(name: str, query: str = "recent conversation"):
    if name not in chat_session["characters"]:
        raise HTTPException(status_code=404, detail="Character not found")
    character = chat_session["characters"][name]
    memories = recall_memories(character, query, top_k=10)
    return {"character": name, "memories": memories}

@app.delete("/characters/{name}/memories")
async def clear_memories(name: str):
    if name not in chat_session["characters"]:
        raise HTTPException(status_code=404, detail="Character not found")
    character = chat_session["characters"][name]
    try:
        chroma_client.delete_collection(character.memory_collection_name)
        logger.info("🧹 Cleared memories for %s", name)
    except Exception:
        pass
    return {"status": "ok"}

@app.post("/reset")
async def reset_session():
    """Clear all conversation transcripts and reset every character's emotion to
    neutral, but KEEP the characters themselves (they now persist on disk)."""
    for character in chat_session["characters"].values():
        character.history.clear()
        character.emotion = EmotionState()
    with _group_lock:
        chat_session["group"] = _fresh_group()
    storage.reset_all()
    logger.info("🔄 Session reset (transcripts cleared, characters kept)")
    return {"status": "ok"}

@app.get("/health")
async def health():
    sd_ok = False
    ollama_ok = False
    try:
        r = requests.get(f"{SD_URL}/sdapi/v1/sd-models", timeout=3)
        sd_ok = r.status_code == 200
    except Exception:
        pass
    try:
        ollama.list()
        ollama_ok = True
    except Exception:
        pass
    return {
        "ollama": ollama_ok,
        "stable_diffusion": sd_ok,
        "tts": voice.tts_available(),
        "stt": voice.stt_available(),
    }

# ---------------------------------------------------------------------------
# Voice (fully offline: Piper TTS + faster-whisper STT)
# ---------------------------------------------------------------------------
@app.get("/voices")
async def get_voices():
    return {"voices": voice.list_voices()}

@app.post("/tts")
async def text_to_speech(payload: TTSPayload):
    text = payload.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="text is required")
    voice_id = payload.voice
    if not voice_id and payload.character_name in chat_session["characters"]:
        voice_id = chat_session["characters"][payload.character_name].voice
    try:
        wav = voice.tts_synth(text, voice_id)
    except Exception as e:
        logger.error("TTS failed: %s", e)
        raise HTTPException(status_code=503, detail=f"TTS unavailable: {e}")
    return Response(content=wav, media_type="audio/wav")

@app.post("/stt")
async def speech_to_text(audio: UploadFile = File(...)):
    data = await audio.read()
    if not data:
        raise HTTPException(status_code=400, detail="empty audio")
    try:
        text = voice.stt_transcribe(data)
    except Exception as e:
        logger.error("STT failed: %s", e)
        raise HTTPException(status_code=503, detail=f"STT unavailable: {e}")
    return {"text": text}

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
