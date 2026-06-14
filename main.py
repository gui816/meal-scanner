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
SITE_PASSWORD = "trilayer"  # fixed password
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
    frequency: str = ""


# ---------------------------------------------------------------------------
# prompt
# ---------------------------------------------------------------------------

# Define response schema for structured JSON output
RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "is_food": {"type": "BOOLEAN", "description": "Whether the image contains a recognizable meal or food plate"},
        "dish_name": {"type": "STRING", "description": "Name of the dish/meal in the user's language, or empty string if not food"},
        "ingredients": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "name": {"type": "STRING", "description": "Ingredient name"},
                    "estimated_amount": {"type": "STRING", "description": "Estimated amount (metric units)"}
                },
                "required": ["name", "estimated_amount"]
            },
            "description": "List of detected ingredients"
        },
        "total_calories_kcal": {"type": "INTEGER", "description": "Total estimated calories in kcal"},
        "protein_g": {"type": "INTEGER", "description": "Protein in grams"},
        "carbs_g": {"type": "INTEGER", "description": "Carbohydrates in grams"},
        "fat_g": {"type": "INTEGER", "description": "Fat in grams"},
        "notes": {"type": "STRING", "description": "Additional nutritional notes in the user's language"},
        "frequency": {"type": "STRING", "description": "Consumption frequency recommendation in user's language: 'Pode consumir diariamente', 'Consumir com moderacao', 'Consumir ocasionalmente', or equivalent"},
    },
    "required": ["is_food", "dish_name", "ingredients", "total_calories_kcal", "protein_g", "carbs_g", "fat_g", "notes", "frequency"]
}

SYSTEM_PROMPT = """Analyse this food photo. Return the analysis in the user's requested language.

If the image DOES contain a recognizable meal, dish, or food plate: set is_food=true and fill all nutritional fields with realistic estimates based on the visible portion.

If the image does NOT contain a recognizable meal or food plate (e.g. landscape, person, document, object, animal, abstract image): set is_food=false and leave all other fields empty/default.

Add a "frequency" field with a consumption recommendation in the user's language when is_food is true. Examples: "Pode consumir diariamente", "Consumir com moderacao", "Consumir ocasionalmente". Base it on the meal's nutritional profile. Use metric units.

{lang}"""


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


def _call_gemini(image: PIL.Image.Image, lang: str = "en") -> MealAnalysis:
    # Add language instruction to prompt
    lang_instruction = {
        "pt": "Responda em português (de Portugal). Use unidades métricas.",
        "en": "Respond in English. Use metric units.",
        "es": "Responde en español. Usa unidades métricas.",
        "fr": "Répondez en français. Utilisez les unités métriques.",
        "de": "Antworte auf Deutsch. Verwende metrische Einheiten.",
        "it": "Rispondi in italiano. Usa unità metriche."
    }
    lang_prompt = lang_instruction.get(lang, lang_instruction["en"])
    prompt = SYSTEM_PROMPT.replace("{lang}", lang_prompt)
    try:
        resp = client.models.generate_content(
            model=MODEL,
            contents=[prompt, image],
            config=types.GenerateContentConfig(
                temperature=0.15,
                max_output_tokens=4096,
                response_mime_type="application/json",
                response_schema=RESPONSE_SCHEMA,
            ),
        )
        raw = resp.text.strip()
        logger.info(f"Gemini raw response ({len(raw)} chars): {raw[:200]}...")
    except Exception as exc:
        err = str(exc)
        logger.error(f"Gemini API call failed: {err[:300]}")
        if "RESOURCE_EXHAUSTED" in err or "quota" in err:
            raise HTTPException(429, f"API quota exceeded: {err[:200]}")
        if "SAFETY" in err or "blocked" in err.lower() or "finish_reason" in err.lower():
            raise HTTPException(422, "BLOCKED")
        raise HTTPException(502, f"Gemini API error: {err[:300]}")

    # With response_mime_type=application/json and response_schema set,
    # Gemini always returns valid JSON. Try to parse it.
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.error(f"Gemini returned invalid JSON despite schema mode: {raw[:300]}")
        # Last resort just in case schema mode fails
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            try:
                data = json.loads(raw[start:end+1])
            except json.JSONDecodeError:
                raise HTTPException(502, f"Gemini returned invalid JSON:\n{raw[:500]}")
        else:
            raise HTTPException(502, f"Gemini returned invalid JSON:\n{raw[:500]}")


    # Validate that the image actually contains food
    dish_name = data.get("dish_name", "")
    ingredients_raw = data.get("ingredients", [])
    total_cal = data.get("total_calories_kcal", 0)
    is_food = data.get("is_food", True)  # default true for backward compat

    # Log what Gemini returned for debugging
    logger.info(f"Gemini analysis: is_food={is_food}, dish='{dish_name}', ingredients={len(ingredients_raw)}, cal={total_cal}")

    # Reject if Gemini explicitly says not food
    if not is_food:
        logger.info("Rejected: Gemini flagged is_food=false")
        raise HTTPException(422, "NOT_FOOD")

    # Reject if dish_name is empty or too generic
    if not dish_name or len(dish_name.strip()) < 3:
        logger.info(f"Rejected: invalid dish_name '{dish_name}'")
        raise HTTPException(422, "NOT_FOOD")

    # Reject if no ingredients detected
    if not ingredients_raw:
        logger.info("Rejected: no ingredients")
        raise HTTPException(422, "NOT_FOOD")

    # Reject if suspiciously low calories with empty dish context
    if total_cal == 0 and len(dish_name) < 5:
        logger.info(f"Rejected: 0 cal with short dish_name '{dish_name}'")
        raise HTTPException(422, "NOT_FOOD")

    # Reject if dish_name suggests it's not food
    non_food_keywords = [
        "no dish", "no food", "unknown", "not a", "not recognized",
        "unrecognizable", "cannot identify", "unable to determine",
        "scenery", "landscape", "portrait", "animal", "document",
        "object", "person", "building", "car", "vehicle"
    ]
    dish_lower = dish_name.lower().strip()
    for kw in non_food_keywords:
        if dish_lower.startswith(kw) or dish_lower == kw:
            logger.info(f"Rejected: dish_name contains non-food keyword '{kw}'")
            raise HTTPException(422, "NOT_FOOD")




    ingredients = [Ingredient(**i) for i in ingredients_raw]
    return MealAnalysis(
        dish_name=dish_name,
        ingredients=ingredients,
        total_calories_kcal=data.get("total_calories_kcal", 0),
        protein_g=data.get("protein_g", 0),
        carbs_g=data.get("carbs_g", 0),
        fat_g=data.get("fat_g", 0),
        notes=data.get("notes", ""),
        frequency=data.get("frequency", ""),
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

app.mount("/static", StaticFiles(directory=Path(__file__).resolve().parent / "static"), name="static")

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
async def analyze(req: Request, file: UploadFile = File(...), lang: str = "en"):
    if not _check_token(req):
        logger.warning(f"Unauthenticated analyze attempt from {req.client.host if req.client else 'unknown'}")
        raise HTTPException(401, "Not authenticated. POST /auth first.")
    if file.content_type not in ALLOWED_TYPES:
        logger.warning(f"Unsupported type: {file.content_type}")
        raise HTTPException(400, f"Unsupported type '{file.content_type}'. Use JPEG, PNG, or WebP.")
    if lang not in ["pt","en","es","fr","de","it"]:
        lang = "en"
    contents = await file.read()
    image = _validate_image(contents, file.filename or "photo")
    try:
        result = _call_gemini(image, lang)
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
