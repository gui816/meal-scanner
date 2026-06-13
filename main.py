"""Meal Scanner — identify ingredients & estimate calories from a plate photo."""

from __future__ import annotations

import io
import json
import logging
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional

import PIL.Image
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from google import genai

# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
MODEL = "gemini-2.0-flash"  # ~$0.075/1M in, $0.30/1M out — balance of quality & cost
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
    estimated_amount: str  # e.g. "150g", "2 units", "15ml"


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

SYSTEM_PROMPT = """You are a professional nutritionist and chef. Analyse the food in this photo.

Return **only** valid JSON — no markdown fences, no extra text.

Schema:
{
  "dish_name": "short descriptive name of the dish",
  "ingredients": [
    {"name": "ingredient name", "estimated_amount": "amount with unit"}
  ],
  "total_calories_kcal": <integer estimate>,
  "protein_g": <integer estimate>,
  "carbs_g": <integer estimate>,
  "fat_g": <integer estimate>,
  "notes": "any relevant observations (e.g. cooking method, sauce, potential allergens)"
}

Rules:
- Be realistic about portion sizes from the photo.
- If you cannot clearly identify the dish, set dish_name to "Unknown dish" and still list visible ingredients.
- Use metric units (g, ml).
- Total calories must be consistent with the listed ingredients + macros."""


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _validate_image(contents: bytes, filename: str) -> PIL.Image.Image:
    if len(contents) > MAX_IMAGE_SIZE:
        raise HTTPException(413, f"Image too large ({len(contents)} bytes). Max {MAX_IMAGE_SIZE} bytes.")
    try:
        img = PIL.Image.open(io.BytesIO(contents))
        img.verify()  # quick integrity check
        img = PIL.Image.open(io.BytesIO(contents))  # re-open after verify
    except Exception as exc:
        raise HTTPException(400, f"Invalid image: {exc}")
    # optional resize for speed / cost
    if max(img.size) > 2048:
        img.thumbnail((2048, 2048), PIL.Image.LANCZOS)
    return img


def _call_gemini(image: PIL.Image.Image) -> MealAnalysis:
    resp = client.models.generate_content(
        model=MODEL,
        contents=[SYSTEM_PROMPT, image],
        config={
            "temperature": 0.2,
            "max_output_tokens": 1024,
        },
    )
    raw = resp.text.strip()
    # Strip markdown fence if the model ignored the instruction
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1]
        raw = raw.rsplit("```", 1)[0]
    raw = raw.strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(502, f"Gemini returned invalid JSON:\n{raw[:500]}") from exc

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

app = FastAPI(
    title="Meal Scanner",
    version="0.1.0",
    description="Upload a photo of a meal and get back ingredients + estimated calories using Gemini Flash.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze", response_model=MealAnalysis)
async def analyze(file: UploadFile = File(...)):
    """Upload a meal photo → receive ingredient list + calorie estimate."""
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(400, f"Unsupported type '{file.content_type}'. Use JPEG, PNG, or WebP.")

    contents = await file.read()
    image = _validate_image(contents, file.filename or "photo")
    result = _call_gemini(image)
    return result


# ---------------------------------------------------------------------------
# CLI convenience
# ---------------------------------------------------------------------------

def cli():
    """Analyse an image from the command line.  Usage: python main.py photo.jpg"""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python main.py <image_path>")
        sys.exit(1)

    path = Path(sys.argv[1])
    if not path.exists():
        print(f"File not found: {path}")
        sys.exit(1)

    image = PIL.Image.open(path)
    result = _call_gemini(image)
    print(json.dumps(asdict(result), indent=2, ensure_ascii=False))


# ---------------------------------------------------------------------------
# entry
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        cli()
    else:
        import uvicorn

        uvicorn.run(
            "main:app",
            host=os.getenv("HOST", "0.0.0.0"),
            port=int(os.getenv("PORT", "8000")),
            reload=True,
        )
