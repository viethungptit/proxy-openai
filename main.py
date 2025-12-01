from fastapi import FastAPI, Request, Header, HTTPException, Response
from fastapi.responses import StreamingResponse, JSONResponse
import os
import httpx
import logging
from dotenv import load_dotenv
from typing import Optional
import json

load_dotenv()
app = FastAPI()

# =========================
# Config
# =========================
PROXY_TOKEN = os.getenv("PROXY_TOKEN")
OPENAI_KEY = os.getenv("OPENAI_KEY")
DEBUG_PROXY = True

# =========================
# Logger
# =========================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("openai-proxy")

# =========================
# Helpers
# =========================
def get_client_ip(request: Request) -> str:
    return request.client.host if request.client else "unknown"

def filter_hop_by_hop_headers(headers):
    hop_by_hop = {
        "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
        "te", "trailers", "transfer-encoding", "upgrade"
    }
    return {k: v for k, v in headers.items() if k.lower() not in hop_by_hop}

def safe_json_response(content: bytes, status_code: int, headers: dict, content_type: str):
    """
    Trả JSONResponse nếu parse được, else fallback về text/plain.
    """
    resp_text = None
    try:
        resp_text = content.decode("utf-8")
    except UnicodeDecodeError:
        if DEBUG_PROXY:
            logger.warning("Response is not UTF-8, returning raw bytes")

    resp_json = None
    if resp_text:
        try:
            resp_json = json.loads(resp_text)
        except json.JSONDecodeError:
            if DEBUG_PROXY:
                logger.warning("Response is not valid JSON, returning text")

    if resp_json is not None:
        return JSONResponse(
            status_code=status_code,
            content=resp_json,
            headers=headers
        )
    else:
        return Response(
            content=resp_text or content,
            status_code=status_code,
            headers=headers,
            media_type="text/plain"
        )

# =========================
# Proxy Route
# =========================
@app.api_route("/v1/{full_path:path}", methods=["GET", "POST", "PUT"])
async def proxy_v1(full_path: str, request: Request, x_proxy_token: Optional[str] = Header(None)):
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
            resp = await client.request(method, target_url, headers=forwarded_headers, content=body_bytes)
            headers = filter_hop_by_hop_headers(resp.headers)
            status_code = resp.status_code
            content_type = resp.headers.get("content-type", "")
            content = resp.content

        except httpx.RequestError as e:
            logger.exception("Request to OpenAI failed: %s", e)
            raise HTTPException(status_code=502, detail="bad_gateway")

    # DEBUG logging
    if DEBUG_PROXY:
        logger.debug("Response status: %s", status_code)
        logger.debug("Response headers: %s", dict(headers))
        logger.debug("Response content (raw bytes, truncated 500): %s", content[:500])

    # Return safely JSON or text
    return safe_json_response(content, status_code, headers, content_type)
