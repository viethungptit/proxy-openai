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
async def proxy_v1(full_path: str, request: Request, x_proxy_token: Optional[str] = Header(None)):
    start = time.perf_counter()
    method = request.method
    client_ip = get_client_ip(request)
    content_length = request.headers.get("content-length", "0")
    token_present = bool(x_proxy_token)
    logger.info(
        "Incoming request %s /v1/%s from %s token_present=%s content_length=%s",
        method, full_path, client_ip, token_present, content_length
    )

    # Auth
    if not x_proxy_token or x_proxy_token != PROXY_TOKEN:
        logger.warning("Unauthorized request %s /v1/%s from %s", method, full_path, client_ip)
        raise HTTPException(status_code=401, detail="unauthorized")

    # Rate limit
    await check_rate_limit(client_ip)

    # Prepare forward
    target_url = f"https://api.openai.com/v1/{full_path}"
    forwarded_headers = {}
    for k, v in request.headers.items():
        if k.lower() not in ("host", "authorization", "x-proxy-token", "content-length", "connection"):
            forwarded_headers[k] = v

    forwarded_headers["Authorization"] = f"Bearer {OPENAI_KEY}"

    # Body
    body_bytes = await request.body()

    timeout = httpx.Timeout(60.0, read=300.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            async with client.stream(
                method,
                target_url,
                headers=forwarded_headers,
                content=body_bytes
            ) as resp:

                headers = filter_hop_by_hop_headers(resp.headers)
                status_code = resp.status_code
                content_type = resp.headers.get("content-type", "")

                if "text/event-stream" in content_type:
                    async def stream_generator():
                        try:
                            async for chunk in resp.aiter_bytes():
                                yield chunk
                        except Exception as e:
                            logger.exception("Error while streaming from OpenAI: %s", e)

                    return StreamingResponse(
                        stream_generator(),
                        status_code=status_code,
                        headers=headers,
                        media_type=content_type
                    )

                content = await resp.aread()

        except httpx.RequestError as e:
            logger.exception("Request to OpenAI failed: %s", e)
            raise HTTPException(status_code=502, detail="bad_gateway")

    if "application/json" in content_type:
        try:
            if isinstance(content, bytes):
                try:
                    resp_text = content.decode("utf-8")
                    logger.info("OpenAI response body (decoded): %s", resp_text)
                except UnicodeDecodeError:
                    logger.warning("Response is not valid UTF-8, logging raw bytes")
                    resp_text = content 
                    logger.info("OpenAI response body (bytes): %s", resp_text)
            try:
                resp_json = resp.json()
                return JSONResponse(
                    status_code=status_code,
                    content=resp_json,
                    headers=headers
                )
            except Exception:
                return Response(
                    content,
                    status_code=status_code,
                    headers=headers,
                    media_type=content_type
                )

        except Exception:
            logger.exception("Unexpected error handling OpenAI response")
            return Response(
                content,
                status_code=status_code,
                headers=headers,
                media_type=content_type
            )
    else:
        return Response(
            content,
            status_code=status_code,
            headers=headers,
            media_type=content_type or "application/octet-stream"
        )

