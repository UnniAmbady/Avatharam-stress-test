# Avatharam-2.2
# Stress-test-3.0
# Ver-8.1  (#Color of buttons - fixed)

import atexit
import json
import os
import time
import subprocess
import re
from pathlib import Path
from typing import Optional, List

import requests
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="Avatharam-2", layout="centered")
st.text("by Krish Ambady")

# ---------------- CSS (unchanged) ----------------
st.markdown(
    """
<style>
  .block-container { padding-top:.6rem; padding-bottom:1rem; }
  iframe { border:none; border-radius:16px; }
  .rowbtn .stButton>button { height:40px; font-size:.95rem; border-radius:12px; }
  div.stChatInput textarea { min-height: 3.4em !important; max-height: 3.8em !important; }

  /* Mic row */
  #microw div[data-testid="stHorizontalBlock"] > div:nth-of-type(1) button {
      background-color: #e74c3c; color: #ffffff; border-color: #e74c3c;
      border-radius: 12px; height: 44px; font-weight: 600;
  }
  #micow div[data-testid="stHorizontalBlock"] > div:nth-of-type(2) button,
  #microw div[data-testid="stHorizontalBlock"] > div:nth-of-type(2) button {
      background-color: #27ae60; color: #ffffff; border-color: #27ae60;
      border-radius: 12px; height: 44px; font-weight: 600;
  }
  #microw div[data-testid="stHorizontalBlock"] > div button:hover { filter: brightness(0.95); }
  #microw div[data-testid="stHorizontalBlock"] > div button:active { transform: translateY(1px); }

  /* Actions row */
  #actrow div[data-testid="stHorizontalBlock"] > div:nth-of-type(1) button {
      background-color: #2980b9; color: #ffffff; border-color: #2980b9;
      border-radius: 12px; height: 48px; font-weight: 600;
  }
  #actrow div[data-testid="stHorizontalBlock"] > div:nth-of-type(2) button {
      background-color: #f39c12; color: #000000; border-color: #f39c12;
      border-radius: 12px; height: 48px; font-weight: 600;
  }
  #actrow div[data-testid="stHorizontalBlock"] > div button:hover { filter: brightness(0.95); }
  #actrow div[data-testid="stHorizontalBlock"] > div button:active { transform: translateY(1px); }
</style>
""",
    unsafe_allow_html=True,
)

# ---------------- Fixed Avatar ----------------
FIXED_AVATAR = {
    "avatar_id": "June_HR_public",
    "default_voice": "68dedac41a9f46a6a4271a95c733823c",
    "normal_preview": "https://files2.heygen.ai/avatar/v3/74447a27859a456c955e01f21ef18216_45620/preview_talk_1.webp",
    "pose_name": "June HR",
    "status": "ACTIVE",
}

# ---------------- Secrets ----------------
SECRETS = st.secrets if "secrets" in dir(st) else {}
HEYGEN_API_KEY = (
    SECRETS.get("HeyGen", {}).get("heygen_api_key")
    or SECRETS.get("heygen", {}).get("heygen_api_key")
    or os.getenv("HEYGEN_API_KEY")
)
OPENAI_API_KEY = (
    SECRETS.get("openai", {}).get("secret_key")
    or os.getenv("OPENAI_API_KEY")
)
if not HEYGEN_API_KEY:
    st.error("Missing HeyGen API key in .streamlit/secrets.toml")
    st.stop()

# ---------------- Endpoints ----------------
BASE = "https://api.heygen.com/v1"
API_STREAM_NEW   = f"{BASE}/streaming.new"
API_CREATE_TOKEN = f"{BASE}/streaming.create_token"
API_STREAM_TASK  = f"{BASE}/streaming.task"
API_STREAM_STOP  = f"{BASE}/streaming.stop"

HEADERS_XAPI = {
    "accept": "application/json",
    "x-api-key": HEYGEN_API_KEY,
    "Content-Type": "application/json",
}
def _headers_bearer(tok: str):
    return {
        "accept": "application/json",
        "Authorization": f"Bearer {tok}",
        "Content-Type": "application/json",
    }

# ---------------- Session State ----------------
ss = st.session_state
ss.setdefault("session_id", None)
ss.setdefault("session_token", None)
ss.setdefault("offer_sdp", None)
ss.setdefault("rtc_config", None)
ss.setdefault("show_sidebar", False)
ss.setdefault("gpt_query", "Hello, welcome.")
ss.setdefault("voice_ready", False)
ss.setdefault("voice_inserted_once", False)
ss.setdefault("bgm_should_play", True)
ss.setdefault("auto_started", False)

# Stress-test memory
ss.setdefault("test_text", "")

# Timer/keepalive state (NEW)
ss.setdefault("stress_active", False)      # set True after Instruction completes
ss.setdefault("next_keepalive_at", 0.0)    # epoch: when to send next "Hi"
ss.setdefault("autorefresh_on", False)     # controls the 2s autorefresh pinger

# ---------------- Debug ----------------
def debug(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

# ---------------- Load Speech.txt ----------------
DEFAULT_INSTRUCTION = (
    "To speak to me, press the speak button, pause a second and then speak. "
    "Once you have spoken press the [Stop] button"
)

def _read_speech_txt() -> Optional[str]:
    """Assumes Speech.txt is in the current working directory."""
    p = Path("Speech.txt")
    if not p.exists() or not p.is_file():
        return None
    try:
        txt = p.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        txt = p.read_text(errors="ignore")
    txt = txt.strip()
    return txt if txt else None

if not ss.get("test_text"):
    loaded = _read_speech_txt()
    if loaded:
        ss.test_text = loaded
        try:
            b = loaded.encode("utf-8")
            debug(f"[stress-load] Speech.txt loaded: chars={len(loaded)}, bytes={len(b)}")
        except Exception:
            debug(f"[stress-load] Speech.txt loaded: chars={len(loaded)} (byte count unavailable)")
    else:
        debug("[stress-load] Speech.txt not found/empty; will use default instruction fallback.")

# ---------------- HTTP helpers ----------------
def _post_xapi(url, payload=None):
    r = requests.post(url, headers=HEADERS_XAPI, data=json.dumps(payload or {}), timeout=120)
    try:
        body = r.json()
    except Exception:
        body = {"_raw": r.text}
    debug(f"[POST x-api] {url} -> {r.status_code}")
    if r.status_code >= 400:
        debug(r.text)
        r.raise_for_status()
    return r.status_code, body

def _post_bearer(url, token, payload=None):
    r = requests.post(url, headers=_headers_bearer(token), data=json.dumps(payload or {}), timeout=600)
    try:
        body = r.json()
    except Exception:
        body = {"_raw": r.text}
    debug(f"[POST bearer] {url} -> {r.status_code}")
    if r.status_code >= 400:
        debug(r.text)
        r.raise_for_status()
    return r.status_code, body

# ---------------- HeyGen helpers ----------------
def new_session(avatar_id: str, voice_id: Optional[str] = None):
    payload = {"avatar_id": avatar_id}
    if voice_id:
        payload["voice_id"] = voice_id
    _, body = _post_xapi(API_STREAM_NEW, payload)
    data = body.get("data") or {}
    sid = data.get("session_id")
    offer_sdp = (data.get("offer") or data.get("sdp") or {}).get("sdp")
    ice2 = data.get("ice_servers2")
    ice1 = data.get("ice_servers")
    if isinstance(ice2, list) and ice2:
        rtc_config = {"iceServers": ice2}
    elif isinstance(ice1, list) and ice1:
        rtc_config = {"iceServers": ice1}
    else:
        rtc_config = {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    if not sid or not offer_sdp:
        raise RuntimeError(f"Missing session_id or offer in response: {body}")
    return {"session_id": sid, "offer_sdp": offer_sdp, "rtc_config": rtc_config}

def create_session_token(session_id: str) -> str:
    _, body = _post_xapi(API_CREATE_TOKEN, {"session_id": session_id})
    tok = (body.get("data") or {}).get("token") or (body.get("data") or {}).get("access_token")
    if not tok:
        raise RuntimeError(f"Missing token in response: {body}")
    return tok

# --- Chunking (<999 chars) ---
MAX_PACKET_CHARS = 999  # NOTE: Empirical limit; ~1 minute of speech at this size.

def _chunk_text(s: str, limit: int = MAX_PACKET_CHARS) -> List[str]:
    s = (s or "").strip()
    if not s:
        return []
    if len(s) <= limit:
        return [s]
    sentences = re.split(r'(?<=[\.\!\?])\s+', s)
    chunks, cur = [], ""
    for sent in sentences:
        if not sent:
            continue
        candidate = (cur + " " + sent).strip() if cur else sent
        if len(candidate) <= limit:
            cur = candidate
        else:
            if cur:
                chunks.append(cur)
            if len(sent) > limit:
                for i in range(0, len(sent), limit):
                    chunks.append(sent[i:i+limit])
                cur = ""
            else:
                cur = sent
    if cur:
        chunks.append(cur)
    return chunks

def _send_task_text(session_id: str, session_token: str, text: str) -> int:
    """Send one task; returns HTTP status code.
    Using task_mode='sync' so 200 means avatar finished speaking this text.
    """
    try:
        b = text.encode("utf-8")
        debug(f"[avatar] send @ {time.strftime('%H:%M:%S')} chars={len(text)}, bytes={len(b)}")
    except Exception:
        debug(f"[avatar] send @ {time.strftime('%H:%M:%S')} chars={len(text)} (bytes n/a)")

    status, body = _post_bearer(
        API_STREAM_TASK,
        session_token,
        {
            "session_id": session_id,
            "task_type": "repeat",
            "task_mode": "sync",   # <-- 200 returned when speech is DONE for this packet
            "text": text,
        },
    )
    debug(f"[avatar] task completed HTTP {status}; body keys={list((body or {}).keys())}")
    return status

def send_text_to_avatar(session_id: str, session_token: str, text: str) -> bool:
    """
    Sends text in <999-char chunks. Returns True when the **final** chunk completed (HTTP 200).
    # NOTE: Keep text length < 999 chars per packet (~1 minute of speech).
    """
    if not text:
        return False
    chunks = _chunk_text(text, MAX_PACKET_CHARS)
    ok = True
    for idx, chunk in enumerate(chunks, 1):
        status = _send_task_text(session_id, session_token, chunk)
        if status != 200:
            ok = False
            debug(f"[avatar] chunk {idx}/{len(chunks)} failed (HTTP {status})")
            break
        else:
            debug(f"[avatar] chunk {idx}/{len(chunks)} OK")
    return ok

def stop_session(session_id: Optional[str], session_token: Optional[str]):
    if not (session_id and session_token):
        return
    try:
        _post_bearer(API_STREAM_STOP, session_token, {"session_id": session_id})
        debug("[stop] session stopped")
    except Exception as e:
        debug(f"[stop_session] {e}")

@atexit.register
def _graceful_shutdown():
    try:
        sid = st.session_state.get("session_id")
        tok = st.session_state.get("session_token")
        if sid and tok:
            stop_session(sid, tok)
    except Exception:
        pass

# ---------------- Audio helpers (unchanged) ----------------
def sniff_mime(b: bytes) -> str:
    try:
        if len(b) >= 12 and b[:4] == b"RIFF" and b[8:12] == b"WAVE": return "audio/wav"
        if b.startswith(b"ID3") or (len(b) > 1 and b[0] == 0xFF and (b[1] & 0xE0) == 0xE0): return "audio/mpeg"
        if b.startswith(b"OggS"): return "audio/ogg"
        if len(b) >= 4 and b[:4] == b"\x1a\x45\xdf\xa3": return "audio/webm"
        if len(b) >= 12 and b[4:8] == b"ftyp": return "audio/mp4"
    except Exception:
        pass
    return "audio/wav"

def _ffmpeg_convert_bytes(inp: bytes, in_ext: str, out_ext: str, ff_args: list) -> tuple[Optional[bytes], bool]:
    try:
        _ = subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        debug("[ffmpeg] not found on PATH")
        return None, False
    try:
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            in_path = Path(td) / f"in{in_ext}"
            out_path = Path(td) / f"out{out_ext}"
            in_path.write_bytes(inp)
            cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(in_path)] + ff_args + [str(out_path)]
            subprocess.run(cmd, check=True)
            out = out_path.read_bytes()
            debug(f"[ffmpeg] converted {in_ext}->{out_ext}, bytes={len(out)}")
            return out, True
    except Exception as e:
        debug(f"[ffmpeg] conversion failed: {repr(e)}")
        return None, False

def prepare_for_soundbar(audio_bytes: bytes, mime: str) -> tuple[bytes, str]:
    if mime in ("audio/webm", "audio/ogg"):
        out, ok = _ffmpeg_convert_bytes(audio_bytes, ".webm" if mime.endswith("webm") else ".ogg", ".wav", ["-ar", "16000", "-ac", "1"])
        debug(f"[soundbar] convert={ok}, final_mime={'audio/wav' if ok else mime}")
        if ok and out: return out, "audio/wav"
        return audio_bytes, mime
    if mime == "audio/mp4":
        debug("[soundbar] pass mp4")
        return audio_bytes, "audio/mp4"
    debug(f"[soundbar] pass-through mime={mime}")
    return audio_bytes, mime

# ---------------- Local ASR helper (unchanged) ----------------
def _save_bytes_tmp(b: bytes, suffix: str) -> str:
    tmp = Path("/tmp") if Path("/tmp").exists() else Path.cwd()
    f = tmp / f"audio_{int(time.time()*1000)}{suffix}"
    f.write_bytes(b); return str(f)

def transcribe_local(audio_bytes: bytes, mime: str) -> str:
    ext = ".wav" if "wav" in mime else ".mp3" if "mp3" in mime else ".webm" if "webm" in mime else ".ogg" if "ogg" in mime else ".m4a"
    fpath = _save_bytes_tmp(audio_bytes, ext)
    try:
        from faster_whisper import WhisperModel
        model = WhisperModel("tiny", device="auto", compute_type="int8")
        segments, _info = model.transcribe(fpath, beam_size=1, language="en")
        txt = " ".join(s.text.strip() for s in segments).strip()
        if txt: return txt
    except Exception as e:
        debug(f"[local asr] faster-whisper error: {repr(e)}")
    try:
        import json as _json
        from vosk import Model, KaldiRecognizer
        model_path = os.getenv("VOSK_MODEL_PATH")
        if model_path and Path(model_path).exists():
            outwav = fpath if fpath.endswith(".wav") else fpath + ".wav"
            try:
                subprocess.run(["ffmpeg", "-y", "-i", fpath, "-ar", "16000", "-ac", "1", outwav], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception:
                outwav = fpath
            import wave
            wf = wave.open(outwav, "rb")
            rec = KaldiRecognizer(Model(model_path), wf.getframerate())
            rec.SetWords(True)
            result = []
            while True:
                data = wf.readframes(4000)
                if len(data) == 0: break
                if rec.AcceptWaveform(data):
                    j = _json.loads(rec.Result()); result.append(j.get("text", ""))
            j = _json.loads(rec.FinalResult()); result.append(j.get("text", ""))
            txt = " ".join(x.strip() for x in result if x).strip()
            if txt: return txt
    except Exception as e:
        debug(f"[local asr] vosk error: {repr(e)}")
    return ""

# ---------------- Header ----------------
cols = st.columns([1, 12, 1])
with cols[0]:
    if st.button("☰", key="btn_trigram_main", help="Open side panel"):
        ss.show_sidebar = not ss.show_sidebar
        debug(f"[ui] sidebar -> {ss.show_sidebar}")

# ---------------- Sidebar (Start/Stop) ----------------
if ss.show_sidebar:
    with st.sidebar:
        st.markdown("### Controls")
        if st.button("Start", key="btn_start_sidebar"):
            if ss.session_id and ss.session_token:
                stop_session(ss.session_id, ss.session_token)
                time.sleep(0.2)
            debug("Step 1: streaming.new")
            created = new_session(FIXED_AVATAR["avatar_id"], FIXED_AVATAR.get("default_voice"))
            sid, offer_sdp, rtc_config = created["session_id"], created["offer_sdp"], created["rtc_config"]
            debug("Step 2: streaming.create_token")
            tok = create_session_token(sid)
            debug("Step 3: sleep 1.0s before viewer")
            time.sleep(1.0)
            ss.session_id, ss.session_token = sid, tok
            ss.offer_sdp, ss.rtc_config = offer_sdp, rtc_config
            ss.bgm_should_play = True
            debug(f"[ready] session_id={sid[:8]}...")
        if st.button("Stop", key="btn_stop_sidebar"):
            stop_session(ss.session_id, ss.session_token)
            ss.session_id = None; ss.session_token = None
            ss.offer_sdp = None; ss.rtc_config = None
            ss.bgm_should_play = False
            # also stop keepalive
            ss.stress_active = False
            ss.autorefresh_on = False
            debug("[stopped] session cleared; keepalive disabled")

# ---------------- Background music ----------------
benhur_path = Path.cwd() / "BenHur-Music.mp3"
if ss.bgm_should_play and benhur_path.exists():
    components.html("<audio id='bgm' src='BenHur-Music.mp3' autoplay loop></audio>", height=0, scrolling=False)
else:
    components.html("<div id='bgm_off'></div>", height=0, scrolling=False)

# ---------------- Auto-start the avatar session ----------------
if not ss.auto_started:
    try:
        debug("[auto-start] initializing session")
        created = new_session(FIXED_AVATAR["avatar_id"], FIXED_AVATAR.get("default_voice"))
        sid, offer_sdp, rtc_config = created["session_id"], created["offer_sdp"], created["rtc_config"]
        tok = create_session_token(sid)
        time.sleep(0.8)
        ss.session_id, ss.session_token = sid, tok
        ss.offer_sdp, ss.rtc_config = offer_sdp, rtc_config
        ss.auto_started = True
        debug(f"[auto-start] session ready id={sid[:8]}...")
    except Exception as e:
        debug(f"[auto-start] failed: {repr(e)}")

# ---------------- Main viewer area ----------------
viewer_candidates = [Path.cwd() / "viewer -Ver-8.1.html", Path.cwd() / "viewer.html"]
viewer_path = next((p for p in viewer_candidates if p.exists()), None)
viewer_loaded = ss.session_id and ss.session_token and ss.offer_sdp

if viewer_loaded and ss.bgm_should_play:
    ss.bgm_should_play = False
    debug("[bgm] stopping background music (viewer ready)")

def _image_compat(url: str, caption: str = ""):
    try:
        st.image(url, caption=caption, use_container_width=True)
    except TypeError:
        try:
            st.image(url, caption=caption, use_column_width=True)
        except TypeError:
            st.image(url, caption=caption)

if viewer_loaded and viewer_path:
    html = (
        viewer_path.read_text(encoding="utf-8")
        .replace("__SESSION_TOKEN__", ss.session_token)
        .replace("__AVATAR_NAME__", FIXED_AVATAR["pose_name"])
        .replace("__SESSION_ID__", ss.session_id)
        .replace("__OFFER_SDP__", json.dumps(ss.offer_sdp)[1:-1])
        .replace("__RTC_CONFIG__", json.dumps(ss.rtc_config or {}))
    )
    components.html(html, height=340, scrolling=False)
else:
    if ss.session_id is None and ss.session_token is None:
        _image_compat(FIXED_AVATAR["normal_preview"], caption=f"{FIXED_AVATAR['pose_name']} ({FIXED_AVATAR['avatar_id']})")

# ---------------- Mic recorder (centered) ----------------
try:
    from streamlit_mic_recorder import mic_recorder
    _HAS_MIC = True
except Exception:
    mic_recorder = None
    _HAS_MIC = False

wav_bytes: Optional[bytes] = None
mime: str = "audio/wav"

with st.container():
    center_cols = st.columns([1, 2, 1])
    with center_cols[1]:
        st.markdown('<div id="microw">', unsafe_allow_html=True)
        audio = mic_recorder(
            start_prompt="Speak",
            stop_prompt="Stop",
            just_once=True,
            use_container_width=True,
            key="mic_recorder_main",
        ) if _HAS_MIC else None
        st.markdown("</div>", unsafe_allow_html=True)

if _HAS_MIC:
    if isinstance(audio, dict) and audio.get("bytes"):
        wav_bytes = audio["bytes"]
        mime = sniff_mime(wav_bytes)
        ss.gpt_query = ""
        ss.voice_inserted_once = False
        ss.voice_ready = True
        debug(f"[mic] received {len(wav_bytes)} bytes, mime={mime}")
    elif isinstance(audio, (bytes, bytearray)) and audio:
        wav_bytes = bytes(audio)
        mime = sniff_mime(wav_bytes)
        ss.gpt_query = ""
        ss.voice_inserted_once = False
        ss.voice_ready = True
        debug(f"[mic] received {len(wav_bytes)} bytes (raw), mime={mime}")

if ss.voice_ready and wav_bytes:
    if not ss.voice_inserted_once:
        transcript_text = ""
        try:
            transcript_text = transcribe_local(wav_bytes, mime)
        except Exception as e:
            debug(f"[voice->text error] {repr(e)}")
        if not transcript_text:
            transcript_text = "(no speech recognized)"
        ss.gpt_query = transcript_text
        ss.voice_inserted_once = True
        debug(f"[voice->editbox] {len(transcript_text)} chars")
    bar_bytes, bar_mime = prepare_for_soundbar(wav_bytes, mime)
    st.audio(bar_bytes, format=bar_mime, autoplay=False)

if ss.voice_ready and ss.voice_inserted_once:
    ss.voice_ready = False

# ---------------- Actions row (Instruction / ChatGPT) ----------------
st.markdown('<div id="actrow">', unsafe_allow_html=True)
col1, col2 = st.columns(2, gap="small")
with col1:
    if st.button("Instruction", key="btn_instruction_main", use_container_width=True):
        if not (ss.session_id and ss.session_token and ss.offer_sdp):
            st.warning("Start a session first.")
        else:
            text_to_send = ss.test_text if ss.test_text else DEFAULT_INSTRUCTION
            # 1) Send long text in chunks; 200 on the final chunk means fully read.
            t0 = time.time()
            ok = send_text_to_avatar(ss.session_id, ss.session_token, text_to_send)
            t1 = time.time()
            debug(f"[timer] long-text send finished ok={ok}; elapsed={t1 - t0:.2f}s")

            # 2) Start/Reset keepalive: first 'Hi' at +50s, then every +60s.
            ss.stress_active = True
            ss.next_keepalive_at = time.time() + 50.0
            ss.autorefresh_on = True
            debug(f"[timer] keepalive armed: next 'Hi' at +50s -> {time.strftime('%H:%M:%S', time.localtime(ss.next_keepalive_at))}")

with col2:
    if st.button("ChatGPT", key="btn_chatgpt_main", use_container_width=True):
        user_text = (ss.get("gpt_query") or "").strip()
        if not user_text:
            debug("[chatgpt] empty user text; skipping]")
        else:
            debug(f"[user->gpt] {len(user_text)} chars")
            url = "https://api.openai.com/v1/chat/completions"
            headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
            payload = {
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": "You are a clear, concise assistant."},
                    {"role": "user", "content": user_text},
                ],
                "temperature": 0.6,
                "max_tokens": 600,
            }
            try:
                r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=60)
                debug(f"[openai] status {r.status_code}")
                body = r.json()
                reply = (body.get("choices", [{}])[0].get("message", {}).get("content") or "").strip()
                if reply:
                    prev = (ss.get("gpt_query") or "").rstrip()
                    joiner = "\n\n" if prev else ""
                    ss.gpt_query = f"{prev}{joiner}Assistant: {reply}"
                    if ss.session_id and ss.session_token:
                        send_text_to_avatar(ss.session_id, ss.session_token, reply)
                else:
                    debug(f"[openai] empty reply: {body}")
            except Exception as e:
                st.error("ChatGPT call failed. See Streamlit logs.")
                debug(f"[openai error] {repr(e)}")
st.markdown("</div>", unsafe_allow_html=True)

# ---------------- Edit box ----------------
ss.gpt_query = st.text_area(
    "Edit message",
    value=ss.get("gpt_query", "Hello, welcome."),
    height=140,
    label_visibility="collapsed",
    key="txt_edit_gpt_query",
)

# ---------------- Autorefresh & Keepalive (NEW) ----------------
def _install_autorefresh(active: bool, interval_ms: int = 2000):
    """Re-run the Streamlit script every interval_ms using a tiny component."""
    if not active:
        return
    components.html(
        f"""
        <script>
        if (window.__stressAutoRefresh) {{
            clearInterval(window.__stressAutoRefresh);
        }}
        window.__stressAutoRefresh = setInterval(function() {{
            window.parent.postMessage({{type: 'streamlit:rerun'}}, '*');
        }}, {interval_ms});
        </script>
        """,
        height=0,
    )

# Arm 2s autorefresh if timer is active
_install_autorefresh(ss.autorefresh_on, 2000)

# On each rerun, if time passed, send "Hi" and schedule next
if ss.stress_active and ss.session_id and ss.session_token and ss.offer_sdp:
    now = time.time()
    if now >= float(ss.next_keepalive_at or 0):
        status = _send_task_text(ss.session_id, ss.session_token, "Hi")
        debug(f"[keepalive] sent @ {time.strftime('%H:%M:%S')} -> HTTP {status}")
        # Next "Hi" in 60 seconds (occupy every 1 minute from last conversation)
        ss.next_keepalive_at = time.time() + 60.0
        debug(f"[timer] next keepalive at {time.strftime('%H:%M:%S', time.localtime(ss.next_keepalive_at))}")
