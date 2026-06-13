"""Meal Scanner — identify ingredients & estimate calories from a plate photo."""

from __future__ import annotations

import hashlib
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

import io
import json
import logging
import os
from dataclasses import dataclass, field, asdict
from functools import wraps
from pathlib import Path
from typing import List, Optional

import PIL.Image
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from google import genai
from google.genai import types

# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
SITE_PASSWORD = os.getenv("SITE_PASSWORD", "trilayer")
PASSWORD_HASH = hashlib.sha256(SITE_PASSWORD.encode()).hexdigest()
MODEL = "gemini-2.5-flash"  # cheapest vision model
MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10 MB
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not set — copy .env.example to .env and fill in your key")

client = genai.Client(api_key=GEMINI_API_KEY)

# ---------------------------------------------------------------------------
# data models
# ---------------------------------------------------------------------------

@dataclass
class Ingredient:
    name: str
    estimated_amount: str


@dataclass
class MealAnalysis:
    dish_name: str
    ingredients: List[Ingredient] = field(default_factory=list)
    total_calories_kcal: int = 0
    protein_g: int = 0
    carbs_g: int = 0
    fat_g: int = 0
    notes: str = ""


# ---------------------------------------------------------------------------
# prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """Analyse this food photo. Return ONLY valid JSON (no markdown, no extra text).

{"dish_name": "...", "ingredients": [{"name": "...", "estimated_amount": "..."}], "total_calories_kcal": 0, "protein_g": 0, "carbs_g": 0, "fat_g": 0, "notes": "..."}

Be realistic about portions. Use metric units."""


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _validate_image(contents: bytes, filename: str) -> PIL.Image.Image:
    logger.info(f"Image received: {filename}, {len(contents)} bytes, type={filename.split('.')[-1] if '.' in filename else 'unknown'}")

    if len(contents) > MAX_IMAGE_SIZE:
        raise HTTPException(413, f"Image too large ({len(contents)} bytes). Max {MAX_IMAGE_SIZE} bytes.")
    try:
        img = PIL.Image.open(io.BytesIO(contents))
        img.verify()
        img = PIL.Image.open(io.BytesIO(contents))
    except Exception as exc:
        raise HTTPException(400, f"Invalid image: {exc}")
    if max(img.size) > 2048:
        img.thumbnail((2048, 2048), PIL.Image.LANCZOS)
    return img


def _call_gemini(image: PIL.Image.Image) -> MealAnalysis:
    try:
        resp = client.models.generate_content(
            model=MODEL,
            contents=[SYSTEM_PROMPT, image],
            config=types.GenerateContentConfig(
                temperature=0.15,
                max_output_tokens=4096,
            ),
        )
        raw = resp.text.strip()
        logger.info(f"Gemini raw response ({len(raw)} chars): {raw[:200]}...")
    except Exception as exc:
        err = str(exc)
        logger.error(f"Gemini API call failed: {err[:300]}")
        if "RESOURCE_EXHAUSTED" in err or "quota" in err:
            raise HTTPException(429, f"API quota exceeded: {err[:200]}")
        raise HTTPException(502, f"Gemini API error: {err[:300]}")

    # Strip markdown code fences if present
    if "```" in raw:
        # Extract JSON between first ``` and last ```
        parts = raw.split("```")
        for p in parts:
            p = p.strip()
            if p.startswith("json"):
                p = p[4:].strip()
            if p.startswith("{") and p.endswith("}"):
                raw = p
                break
        else:
            # fallback: just strip the fences
            raw = raw.replace("```json", "").replace("```", "").strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # Last resort: find first { and last }
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            raw = raw[start:end+1]
            try:
                data = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise HTTPException(502, f"Gemini returned invalid JSON:\n{raw[:500]}")
        else:
            raise HTTPException(502, f"Gemini returned invalid JSON:\n{raw[:500]}")



    ingredients = [Ingredient(**i) for i in data.get("ingredients", [])]
    return MealAnalysis(
        dish_name=data.get("dish_name", "Unknown dish"),
        ingredients=ingredients,
        total_calories_kcal=data.get("total_calories_kcal", 0),
        protein_g=data.get("protein_g", 0),
        carbs_g=data.get("carbs_g", 0),
        fat_g=data.get("fat_g", 0),
        notes=data.get("notes", ""),
    )


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="Meal Scanner", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Auth helpers ──────────────────────────────────────────────────────

TOKEN_HEADER = "x-scan-auth"


def _make_token() -> str:
    return hashlib.sha256(f"{PASSWORD_HASH}:meal-scan-secret".encode()).hexdigest()


def _check_token(req: Request):
    return req.headers.get(TOKEN_HEADER) == _make_token()


# ── API routes ────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/auth")
def auth(body: dict):
    pwd = body.get("password", "")
    if hashlib.sha256(pwd.encode()).hexdigest() == PASSWORD_HASH:
        return {"token": _make_token()}
    raise HTTPException(401, "Wrong password")


@app.post("/analyze", response_model=MealAnalysis)
async def analyze(req: Request, file: UploadFile = File(...)):
    if not _check_token(req):
        logger.warning(f"Unauthenticated analyze attempt from {req.client.host if req.client else 'unknown'}")
        raise HTTPException(401, "Not authenticated. POST /auth first.")
    if file.content_type not in ALLOWED_TYPES:
        logger.warning(f"Unsupported type: {file.content_type}")
        raise HTTPException(400, f"Unsupported type '{file.content_type}'. Use JPEG, PNG, or WebP.")
    contents = await file.read()
    image = _validate_image(contents, file.filename or "photo")
    try:
        result = _call_gemini(image)
        logger.info(f"Analyze success: {result.dish_name}, {result.total_calories_kcal} kcal, {len(result.ingredients)} ingredients")
        return result
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Unexpected error in analyze: {exc}", exc_info=True)
        raise HTTPException(500, f"Internal error: {str(exc)[:200]}")


# ── Frontend ──────────────────────────────────────────────────────

HTML_PAGE = Path(__file__).resolve().parent / "static" / "index.html"
if not HTML_PAGE.exists():
    HTML_PAGE = "<h1>Meal Scanner</h1><p>Install error: static/index.html missing</p>"
else:
    HTML_PAGE = HTML_PAGE.read_text(encoding="utf-8")


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML_PAGE
