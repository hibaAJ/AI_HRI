import os
import json
import socket
import datetime
import ipaddress
import asyncio
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from sentence_transformers import SentenceTransformer

# ── AI state ─────────────────────────────────────────────────
qa_pairs: list[dict] = []
qa_embeddings: np.ndarray | None = None
model: SentenceTransformer | None = None

THRESHOLD = 0.42
FALLBACK = "Sorry, I don't have the context for that. Please repeat your question to the operator."


def find_answer(query: str) -> str:
    if model is None or qa_embeddings is None:
        return FALLBACK
    q_emb = model.encode([query], normalize_embeddings=True)
    scores = (qa_embeddings @ q_emb.T).flatten()
    best = int(scores.argmax())
    if scores[best] < THRESHOLD:
        return FALLBACK
    return qa_pairs[best]["a"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, qa_embeddings, qa_pairs
    print("Loading CCI knowledge base...")
    with open("cci_data.json", encoding="utf-8") as f:
        qa_pairs = json.load(f)
    model = SentenceTransformer("all-MiniLM-L6-v2")
    questions = [item["q"] for item in qa_pairs]
    qa_embeddings = model.encode(questions, normalize_embeddings=True)
    print(f"Loaded {len(qa_pairs)} Q&A pairs. Ready.")
    yield


# ── App & clients ─────────────────────────────────────────────
app = FastAPI(lifespan=lifespan)

clients: dict[str, WebSocket | None] = {"ground": None, "tablet": None}
clients_lock = asyncio.Lock()


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
    print(f"Generating SSL certificate for localhost and {local_ip}...")

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
    print("SSL certificate generated.")


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
    other_connected = clients.get(other_role) is not None
    try:
        await websocket.send_text(json.dumps({
            "type": "peer_status",
            "role": other_role,
            "connected": other_connected,
        }))
    except Exception:
        pass

    try:
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)
            msg["from"] = role

            if msg.get("type") == "ai_query" and role == "tablet":
                # AI handles the question
                answer = find_answer(msg["text"])
                try:
                    await websocket.send_text(json.dumps({
                        "type": "ai_response",
                        "text": answer,
                    }))
                except Exception:
                    pass
                # Log the exchange to ground station
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
                # Relay to the other side (operator overrides, etc.)
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

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        ssl_certfile="cert.pem",
        ssl_keyfile="key.pem",
    )
