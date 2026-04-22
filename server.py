import os
import json
import shutil
import socket
import datetime
import ipaddress
import asyncio
import tempfile
from contextlib import asynccontextmanager
import glob as _glob

# ── Ensure ffmpeg is findable (winget puts it outside the default PATH) ───────
def _add_ffmpeg_to_path():
    if shutil.which("ffmpeg"):
        return
    winget_base = os.path.join(os.environ.get("LOCALAPPDATA", ""), "Microsoft", "WinGet", "Packages")
    for exe in _glob.glob(os.path.join(winget_base, "Gyan.FFmpeg*", "**", "bin", "ffmpeg.exe"), recursive=True):
        os.environ["PATH"] = os.path.dirname(exe) + os.pathsep + os.environ.get("PATH", "")
        print(f"ffmpeg found at: {exe}")
        return
    print("WARNING: ffmpeg not found in PATH or WinGet packages. Whisper transcription may fail.")

_add_ffmpeg_to_path()

import ollama as ollama_client
from faster_whisper import WhisperModel
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

# ── Whisper ───────────────────────────────────────────────────
WHISPER_PROMPT = (
    "University of Sharjah, UOS, CCI, College of Computing and Informatics, "
    "informatics, computer science, computer engineering, ABET, cybersecurity, "
    "biomedical informatics, Sharjah, UAE"
)
whisper_model: WhisperModel | None = None

# ── Ollama ────────────────────────────────────────────────────
SYSTEM_PROMPT = ""
FALLBACK = "Sorry, I don't have the context for that. Please repeat your question to the operator."
MODEL = "llama3.2"


def build_system_prompt(qa_pairs: list[dict]) -> str:
    kb = "\n\n".join(f"Q: {item['q']}\nA: {item['a']}" for item in qa_pairs)
    return f"""You are CCI GuideBot, a voice assistant specifically for the College of Computing and Informatics (CCI) at the University of Sharjah.

SPEECH RECOGNITION NOTE:
Input comes from a microphone. "Sharjah" may be misheard as "Georgia", "Sasha", "Sure", etc. "UOS" means University of Sharjah. "CCI" may be heard as "CC I" or "Sisi". Use context to interpret correctly.

RULES:
- You ONLY know about CCI. You do not know about other colleges, universities, or topics.
- Use ONLY the knowledge base below. Do not make up information.
- Keep answers short: 2-4 sentences max. This is spoken aloud, so no bullet points or markdown.
- If someone asks about the University of Sharjah broadly, say: "I can only speak about the College of Computing and Informatics at the University of Sharjah." Then answer with CCI info if relevant.
- If the question has nothing to do with CCI at all, respond with EXACTLY this and nothing else:
  {FALLBACK}

KNOWLEDGE BASE:
{kb}"""


async def find_answer(query: str) -> str:
    try:
        client = ollama_client.AsyncClient()
        response = await client.chat(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
        )
        return response.message.content.strip()
    except Exception as e:
        print(f"Ollama error: {e}")
        return FALLBACK


# ── Startup ───────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global whisper_model, SYSTEM_PROMPT
    print("Loading Whisper model...")
    whisper_model = WhisperModel("base", device="cpu", compute_type="int8")
    print("Loading CCI knowledge base...")
    with open("cci_data.json", encoding="utf-8") as f:
        qa_pairs = json.load(f)
    SYSTEM_PROMPT = build_system_prompt(qa_pairs)
    print(f"Ready. {len(qa_pairs)} Q&A pairs loaded.")
    yield


# ── App ───────────────────────────────────────────────────────
app = FastAPI(lifespan=lifespan)

clients: dict[str, WebSocket | None] = {"ground": None, "tablet": None}
clients_lock = asyncio.Lock()


@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    data = await audio.read()
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as f:
        f.write(data)
        tmp_path = f.name
    try:
        segments, _ = whisper_model.transcribe(
            tmp_path,
            language="en",
            initial_prompt=WHISPER_PROMPT,
        )
        text = " ".join(seg.text.strip() for seg in segments).strip()
        return JSONResponse({"text": text})
    finally:
        os.unlink(tmp_path)


# ── Helpers ───────────────────────────────────────────────────
def get_local_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def generate_ssl_cert():
    if os.path.exists("cert.pem") and os.path.exists("key.pem"):
        return
    from cryptography import x509
    from cryptography.x509.oid import NameOID
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa

    local_ip = get_local_ip()
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = issuer = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "HRI Local")])
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.datetime.utcnow())
        .not_valid_after(datetime.datetime.utcnow() + datetime.timedelta(days=365))
        .add_extension(
            x509.SubjectAlternativeName([
                x509.DNSName("localhost"),
                x509.IPAddress(ipaddress.IPv4Address("127.0.0.1")),
                x509.IPAddress(ipaddress.IPv4Address(local_ip)),
            ]),
            critical=False,
        )
        .sign(key, hashes.SHA256())
    )
    with open("cert.pem", "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))
    with open("key.pem", "wb") as f:
        f.write(key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.TraditionalOpenSSL,
            serialization.NoEncryption(),
        ))
    print(f"SSL certificate generated for {local_ip}.")


async def notify_peer_status(changed_role: str, connected: bool):
    other_role = "tablet" if changed_role == "ground" else "ground"
    other_ws = clients.get(other_role)
    if other_ws:
        try:
            await other_ws.send_text(json.dumps({
                "type": "peer_status",
                "role": changed_role,
                "connected": connected,
            }))
        except Exception:
            pass


# ── WebSocket ─────────────────────────────────────────────────
@app.websocket("/ws/{role}")
async def websocket_endpoint(websocket: WebSocket, role: str):
    if role not in ("ground", "tablet"):
        await websocket.close(code=1008)
        return

    await websocket.accept()

    async with clients_lock:
        existing = clients[role]
        if existing:
            try:
                await existing.close()
            except Exception:
                pass
        clients[role] = websocket

    print(f"[{role}] connected")
    await notify_peer_status(role, True)

    other_role = "tablet" if role == "ground" else "ground"
    try:
        await websocket.send_text(json.dumps({
            "type": "peer_status",
            "role": other_role,
            "connected": clients.get(other_role) is not None,
        }))
    except Exception:
        pass

    try:
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)
            msg["from"] = role

            if msg.get("type") == "ai_query" and role == "tablet":
                answer = await find_answer(msg["text"])
                try:
                    await websocket.send_text(json.dumps({"type": "ai_response", "text": answer}))
                except Exception:
                    pass
                ground_ws = clients.get("ground")
                if ground_ws:
                    try:
                        await ground_ws.send_text(json.dumps({
                            "type": "ai_log",
                            "question": msg["text"],
                            "answer": answer,
                        }))
                    except Exception:
                        pass
            else:
                other_ws = clients.get(other_role)
                if other_ws:
                    try:
                        await other_ws.send_text(json.dumps(msg))
                    except Exception:
                        pass

    except WebSocketDisconnect:
        async with clients_lock:
            if clients[role] is websocket:
                clients[role] = None
        print(f"[{role}] disconnected")
        await notify_peer_status(role, False)


app.mount("/", StaticFiles(directory="static", html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    generate_ssl_cert()
    local_ip = get_local_ip()
    print("\n" + "=" * 50)
    print("  CCI GuideBot Server")
    print("=" * 50)
    print(f"  Ground Station : https://localhost:8000/ground_station.html")
    print(f"  Tablet         : https://{local_ip}:8000/tablet.html")
    print("=" * 50 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000, ssl_certfile="cert.pem", ssl_keyfile="key.pem")
