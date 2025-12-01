from fastapi import FastAPI, Request, Header, HTTPException, Response
from fastapi.responses import StreamingResponse, JSONResponse
import os
import httpx
import asyncio
import logging
import time
from dotenv import load_dotenv
from typing import Optional
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("openai-proxy")

if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)
logger.propagate = False

# Load environment variables
load_dotenv()

# Config from env
OPENAI_KEY = os.getenv("OPENAI_KEY")
PROXY_TOKEN = os.getenv("PROXY_TOKEN")
PORT = int(os.getenv("PORT") or 3000)
RATE_WINDOW_MS = int(os.getenv("RATE_WINDOW_MS") or 60_000)
RATE_MAX = int(os.getenv("RATE_MAX") or 60)
DEBUG_PROXY = os.getenv("DEBUG_PROXY", "false").lower() == "true"

if not OPENAI_KEY:
    logger.error("OPENAI_KEY not set. Exiting.")
    raise RuntimeError("OPENAI_KEY not configured")
if not PROXY_TOKEN:
    logger.error("PROXY_TOKEN not set. Exiting.")
    raise RuntimeError("PROXY_TOKEN not configured")

app = FastAPI(title="OpenAI Proxy", version="0.1")

# In-memory rate limiter
clients_rate = {}  # ip -> [window_start:datetime, count:int]
rate_lock = asyncio.Lock()


def get_client_ip(request: Request) -> str:
    xff = request.headers.get("x-forwarded-for")
    if xff:
        return xff.split(",")[0].strip()
    client = request.client
    return client.host if client else "unknown"


async def check_rate_limit(ip: str):
    async with rate_lock:
        now = datetime.utcnow()
        entry = clients_rate.get(ip)
        if not entry:
            clients_rate[ip] = [now, 1]
            return
        window_start, count = entry
        if now - window_start > timedelta(milliseconds=RATE_WINDOW_MS):
            clients_rate[ip] = [now, 1]
            return
        if count >= RATE_MAX:
            raise HTTPException(status_code=429, detail="rate_limit_exceeded")
        clients_rate[ip][1] += 1


def filter_hop_by_hop_headers(headers: httpx.Headers):
    hop_by_hop = {
        "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
        "te", "trailers", "transfer-encoding", "upgrade", "content-encoding" 
    }
    return {k: v for k, v in headers.items() if k.lower() not in hop_by_hop}



@app.get("/_health")
async def health():
    return {"ok": True}


@app.api_route("/v1/{full_path:path}", methods=["GET", "POST", "PUT"])
async def proxy_v1_debug(full_path: str, request: Request, x_proxy_token: Optional[str] = Header(None)):
    method = request.method
    client_ip = get_client_ip(request)
    token_present = bool(x_proxy_token)

    logger.info(
        "Incoming request %s /v1/%s from %s token_present=%s",
        method, full_path, client_ip, token_present
    )

    # Auth
    if not x_proxy_token or x_proxy_token != PROXY_TOKEN:
        logger.warning("Unauthorized request %s /v1/%s from %s", method, full_path, client_ip)
        raise HTTPException(status_code=401, detail="unauthorized")

    # Rate limit
    await check_rate_limit(client_ip)

    # Forward setup
    target_url = f"https://api.openai.com/v1/{full_path}"
    forwarded_headers = {
        k: v for k, v in request.headers.items()
        if k.lower() not in ("host", "authorization", "x-proxy-token", "content-length", "connection")
    }
    forwarded_headers["Authorization"] = f"Bearer {OPENAI_KEY}"

    body_bytes = await request.body()
    timeout = httpx.Timeout(60.0, read=300.0)

    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            async with client.stream(method, target_url, headers=forwarded_headers, content=body_bytes) as resp:
                headers = filter_hop_by_hop_headers(resp.headers)
                status_code = resp.status_code
                content_type = resp.headers.get("content-type", "")

                # Streaming
                if "text/event-stream" in content_type:
                    async def stream_generator():
                        async for chunk in resp.aiter_bytes():
                            if DEBUG_PROXY:
                                logger.debug("Streaming chunk (%d bytes): %s", len(chunk), chunk[:200])
                            yield chunk

                    return StreamingResponse(
                        stream_generator(),
                        status_code=status_code,
                        headers=headers,
                        media_type=content_type
                    )

                # Non-streaming: read all
                content = await resp.aread()

        except httpx.RequestError as e:
            logger.exception("Request to OpenAI failed: %s", e)
            raise HTTPException(status_code=502, detail="bad_gateway")

    # DEBUG logging
    if DEBUG_PROXY:
        logger.debug("Response status: %s", status_code)
        logger.debug("Response headers: %s", dict(headers))
        logger.debug("Response content (raw bytes, truncated 500): %s", content[:500])

    # Decode UTF-8 if possible
    try:
        resp_text = content.decode("utf-8")
        if DEBUG_PROXY:
            logger.debug("Response content decoded UTF-8 (truncated 500): %s", resp_text[:500])
    except UnicodeDecodeError:
        resp_text = str(content)
        if DEBUG_PROXY:
            logger.debug("Response content is not UTF-8, showing as str (truncated 500): %s", resp_text[:500])

    # Try parse JSON
    resp_json = None
    try:
        import json
        resp_json = json.loads(resp_text)
    except Exception:
        if DEBUG_PROXY:
            logger.warning("Response is not valid JSON, returning raw content")

    # Return response
    if resp_json is not None:
        return JSONResponse(
            status_code=status_code,
            content=resp_json,
            headers=headers
        )
    else:
        return Response(
            content=resp_text,
            status_code=status_code,
            headers=headers,
            media_type=content_type or "application/octet-stream"
        )
