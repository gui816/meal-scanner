"""Meal Scanner — identify ingredients & estimate calories from a plate photo."""

from __future__ import annotations

import hashlib
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

# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
SITE_PASSWORD = os.getenv("SITE_PASSWORD", "scanner")
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
  "notes": "any relevant observations"
}

Rules:
- Be realistic about portion sizes.
- If you cannot clearly identify the dish, set dish_name to "Unknown dish".
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
        img.verify()
        img = PIL.Image.open(io.BytesIO(contents))
    except Exception as exc:
        raise HTTPException(400, f"Invalid image: {exc}")
    if max(img.size) > 2048:
        img.thumbnail((2048, 2048), PIL.Image.LANCZOS)
    return img


def _call_gemini(image: PIL.Image.Image) -> MealAnalysis:
    resp = client.models.generate_content(
        model=MODEL,
        contents=[SYSTEM_PROMPT, image],
        config={"temperature": 0.15, "max_output_tokens": 1024},
    )
    raw = resp.text.strip()
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
        raise HTTPException(401, "Not authenticated. POST /auth first.")
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(400, f"Unsupported type '{file.content_type}'. Use JPEG, PNG, or WebP.")
    contents = await file.read()
    image = _validate_image(contents, file.filename or "photo")
    return _call_gemini(image)


# ── Frontend ──────────────────────────────────────────────────────────

HTML_PAGE = """<!DOCTYPE html>
<html lang="pt">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1.0"/>
<title>Meal Scanner</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
:root{--bg:#0a0b14;--bg2:#12141f;--text:#eef1fa;--sub:#6b7394;--border:rgba(255,255,255,.07);--v:#7c3aed;--b:#3b82f6;--c:#06b6d4;--g:#10b981}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',system-ui,sans-serif;background:var(--bg);color:var(--text);min-height:100vh;display:flex;align-items:center;justify-content:center;padding:1rem}
/* GATE */
#gate{position:fixed;inset:0;z-index:999;display:flex;align-items:center;justify-content:center;background:var(--bg)}
.gate-card{text-align:center;width:100%;max-width:360px;padding:2rem}
.gate-logo{font-size:2.8rem;margin-bottom:.2rem}
.gate-title{font-size:1.4rem;font-weight:700;margin-bottom:.3rem}
.gate-sub{color:var(--sub);font-size:.85rem;margin-bottom:2rem}
.gate-card input{width:100%;padding:.8rem 1rem;margin-bottom:.7rem;background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.09);border-radius:10px;color:var(--text);font-size:.95rem;outline:none;font-family:inherit}
.gate-card input:focus{border-color:rgba(124,58,237,.5);background:rgba(124,58,237,.07)}
.gate-btn{width:100%;padding:.85rem;background:var(--v);color:#fff;border:none;border-radius:10px;font-size:.95rem;font-weight:700;cursor:pointer;font-family:inherit;transition:background .2s}
.gate-btn:hover{background:#6d28d9}
.gate-err{color:#fb7185;font-size:.82rem;margin-top:.7rem;display:none}
/* APP */
#app{display:none;width:100%;max-width:600px;margin:0 auto}
header{text-align:center;margin-bottom:2rem}
header h1{font-size:1.8rem;font-weight:800;letter-spacing:-.02em}
header h1 span{background:linear-gradient(135deg,#c4b5fd,#7dd3fc);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
header p{color:var(--sub);font-size:.85rem;margin-top:.3rem}
/* UPLOAD ZONE */
.upload-zone{border:2px dashed rgba(255,255,255,.12);border-radius:20px;padding:3rem 2rem;text-align:center;cursor:pointer;transition:all .3s;position:relative;background:rgba(255,255,255,.02)}
.upload-zone:hover,.upload-zone.dragover{border-color:rgba(124,58,237,.5);background:rgba(124,58,237,.06);transform:scale(1.01)}
.upload-zone.has-img{border-color:var(--g);border-style:solid;background:rgba(16,185,129,.04)}
.uz-icon{font-size:3.5rem;margin-bottom:.8rem}
.uz-title{font-weight:700;font-size:1rem;margin-bottom:.3rem}
.uz-sub{color:var(--sub);font-size:.82rem}
.uz-file{display:none;margin-top:.8rem;color:var(--g);font-size:.85rem;font-weight:600}
.uz-preview{display:none;margin-top:1.2rem;max-width:100%;max-height:260px;border-radius:12px;object-fit:cover}
.btn-analyze{display:none;margin:1.2rem auto 0;padding:.85rem 2.5rem;background:linear-gradient(135deg,var(--v),var(--b));color:#fff;border:none;border-radius:12px;font-size:1rem;font-weight:700;cursor:pointer;font-family:inherit;transition:all .2s;box-shadow:0 6px 20px rgba(124,58,237,.3)}
.btn-analyze:hover{transform:translateY(-2px);box-shadow:0 12px 30px rgba(124,58,237,.5)}
.btn-analyze:disabled{opacity:.5;cursor:not-allowed;transform:none}
/* LOADING */
.loading{display:none;text-align:center;margin:2rem 0}
.spinner{width:40px;height:40px;border:4px solid rgba(255,255,255,.08);border-top-color:var(--v);border-radius:50%;animation:spin .8s linear infinite;margin:0 auto .8rem}
@keyframes spin{to{transform:rotate(360deg)}}
.loading p{color:var(--sub);font-size:.85rem}
/* RESULTS */
#results{display:none;margin-top:2rem;animation:fadeIn .4s ease}
@keyframes fadeIn{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:translateY(0)}}
.result-header{display:flex;align-items:center;gap:.7rem;margin-bottom:1rem;padding-bottom:.8rem;border-bottom:1px solid var(--border)}
.result-header h2{font-size:1.3rem;font-weight:700}
.result-header .cal{background:rgba(16,185,129,.12);color:var(--g);padding:.25rem .9rem;border-radius:20px;font-size:.8rem;font-weight:700}
.macro-row{display:flex;gap:.5rem;flex-wrap:wrap;margin-bottom:1.5rem}
.macro{flex:1;min-width:80px;background:var(--bg2);border:1px solid var(--border);border-radius:12px;padding:.8rem;text-align:center}
.macro-num{font-size:1.1rem;font-weight:800;background:linear-gradient(135deg,#c4b5fd,#7dd3fc);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.macro-lbl{font-size:.72rem;color:var(--sub);margin-top:2px}
.ingredients{border-radius:14px;overflow:hidden;border:1px solid var(--border)}
.ing-title{padding:.8rem 1.2rem;background:var(--bg2);font-size:.78rem;font-weight:700;text-transform:uppercase;letter-spacing:.06em;color:var(--sub)}
.ing-list{list-style:none}
.ing-item{display:flex;justify-content:space-between;padding:.65rem 1.2rem;border-top:1px solid var(--border);font-size:.88rem}
.ing-item:last-child{border-bottom:none}
.ing-amount{color:var(--sub)}
.notes{margin-top:1rem;padding:1rem 1.2rem;background:rgba(6,182,212,.06);border:1px solid rgba(6,182,212,.15);border-radius:12px;font-size:.85rem;color:rgba(238,241,250,.7);line-height:1.6}
.reset-btn{display:inline-block;margin-top:1.2rem;background:none;border:1px solid var(--border);color:var(--sub);padding:.5rem 1.2rem;border-radius:8px;font-size:.82rem;cursor:pointer;font-family:inherit;transition:all .2s}
.reset-btn:hover{background:rgba(255,255,255,.05);color:var(--text)}
footer{text-align:center;margin-top:2.5rem;color:rgba(255,255,255,.12);font-size:.72rem}
</style>
</head>
<body>

<!-- GATE -->
<div id="gate">
  <div class="gate-card">
    <div class="gate-logo">🍽️</div>
    <div class="gate-title">Meal Scanner</div>
    <div class="gate-sub">Enter password to access</div>
    <input type="password" id="gp" placeholder="Password" autocomplete="off" onkeydown="if(event.key==='Enter')doAuth()"/>
    <button class="gate-btn" onclick="doAuth()">Enter →</button>
    <div class="gate-err" id="ge">Wrong password. Try again.</div>
  </div>
</div>

<!-- APP -->
<div id="app">
  <header>
    <h1>🍽️ <span>Meal Scanner</span></h1>
    <p>Upload a photo of your meal — get ingredients & estimated calories</p>
  </header>

  <div class="upload-zone" id="dropZone" onclick="document.getElementById('fileInput').click()">
    <div class="uz-icon">📸</div>
    <div class="uz-title">Drop your photo here</div>
    <div class="uz-sub">or click to browse · JPEG, PNG, WebP</div>
    <div class="uz-file" id="fileName"></div>
    <img class="uz-preview" id="preview" alt="preview"/>
    <input type="file" id="fileInput" accept="image/jpeg,image/png,image/webp" style="display:none" onchange="handleFile(this)"/>
  </div>

  <button class="btn-analyze" id="analyzeBtn" onclick="analyze()">🔬 Analyze meal</button>

  <div class="loading" id="loading">
    <div class="spinner"></div>
    <p>Analysing your meal...</p>
  </div>

  <div id="results">
    <div class="result-header">
      <h2 id="dishName"></h2>
      <div class="cal" id="calBadge"></div>
    </div>
    <div class="macro-row" id="macroRow"></div>
    <div class="ingredients">
      <div class="ing-title">🥘 Ingredients</div>
      <ul class="ing-list" id="ingList"></ul>
    </div>
    <div class="notes" id="notes"></div>
    <button class="reset-btn" onclick="reset()">← Scan another meal</button>
  </div>

  <footer>Powered by Gemini Flash</footer>
</div>

<script>
let TOKEN = null;

// ── Auth ──
async function doAuth(){
  const pwd = document.getElementById('gp').value;
  const err = document.getElementById('ge');
  try {
    const r = await fetch('/auth', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({password:pwd})});
    if(!r.ok) { err.style.display='block'; return; }
    const d = await r.json();
    TOKEN = d.token;
    sessionStorage.setItem('ms_token', TOKEN);
    document.getElementById('gate').style.display='none';
    document.getElementById('app').style.display='block';
  } catch(e) { err.style.display='block'; }
}
// Auto auth
window.addEventListener('DOMContentLoaded', ()=>{
  const t = sessionStorage.getItem('ms_token');
  if(t) { TOKEN=t; document.getElementById('gate').style.display='none'; document.getElementById('app').style.display='block'; }
});

// ── File handling ──
function handleFile(input){
  const f = input.files[0];
  if(!f) return;
  const zone = document.getElementById('dropZone');
  zone.classList.add('has-img');
  document.getElementById('fileName').textContent = '📄 ' + f.name;
  document.getElementById('fileName').style.display='block';
  const reader = new FileReader();
  reader.onload = e => {
    const p = document.getElementById('preview');
    p.src = e.target.result;
    p.style.display='block';
    document.querySelector('.uz-icon').style.display='none';
    document.querySelector('.uz-title').textContent='Ready to scan!';
    document.querySelector('.uz-sub').style.display='none';
  };
  reader.readAsDataURL(f);
  document.getElementById('analyzeBtn').style.display='block';
}

// ── Drag & drop ──
const dz = document.getElementById('dropZone');
dz.addEventListener('dragover', e => { e.preventDefault(); dz.classList.add('dragover'); });
dz.addEventListener('dragleave', () => dz.classList.remove('dragover'));
dz.addEventListener('drop', e => {
  e.preventDefault();
  dz.classList.remove('dragover');
  const f = e.dataTransfer.files[0];
  if(f) { document.getElementById('fileInput').files = e.dataTransfer.files; handleFile(document.getElementById('fileInput')); }
});

// ── Analyze ──
async function analyze(){
  const fileInput = document.getElementById('fileInput');
  if(!fileInput.files[0]) return;
  const btn = document.getElementById('analyzeBtn');
  btn.disabled = true; btn.textContent = '⏳ Analysing...';
  document.getElementById('loading').style.display='block';
  document.getElementById('results').style.display='none';

  const fd = new FormData();
  fd.append('file', fileInput.files[0]);

  try {
    const r = await fetch('/analyze', {method:'POST', headers:{'x-scan-auth': TOKEN}, body: fd});
    if(r.status === 401) { sessionStorage.removeItem('ms_token'); location.reload(); return; }
    const data = await r.json();
    showResults(data);
  } catch(e) {
    document.getElementById('loading').innerHTML = '<p style="color:#fb7185">❌ Error analysing image. Try again.</p>';
  }
  btn.disabled = false; btn.textContent = '🔬 Analyze meal';
  document.getElementById('loading').style.display='none';
}

function showResults(d){
  document.getElementById('dishName').textContent = d.dish_name;
  document.getElementById('calBadge').textContent = '🔥 ' + d.total_calories_kcal + ' kcal';

  document.getElementById('macroRow').innerHTML = [
    {l:'Protein', v:d.protein_g, u:'g'},
    {l:'Carbs', v:d.carbs_g, u:'g'},
    {l:'Fat', v:d.fat_g, u:'g'}
  ].map(m => '<div class="macro"><div class="macro-num">'+m.v+'</div><div class="macro-lbl">'+m.u+' ' + m.l+'</div></div>').join('');

  document.getElementById('ingList').innerHTML = d.ingredients.map(i =>
    '<li class="ing-item"><span>'+i.name+'</span><span class="ing-amount">'+i.estimated_amount+'</span></li>'
  ).join('');

  document.getElementById('notes').textContent = d.notes;
  document.getElementById('results').style.display='block';
  document.getElementById('results').scrollIntoView({behavior:'smooth'});
}

function reset(){
  document.getElementById('results').style.display='none';
  document.getElementById('analyzeBtn').style.display='none';
  document.getElementById('fileInput').value = '';
  document.getElementById('preview').style.display='none';
  document.getElementById('preview').src = '';
  document.getElementById('fileName').style.display='none';
  const zone = document.getElementById('dropZone');
  zone.classList.remove('has-img');
  document.querySelector('.uz-icon').style.display='block';
  document.querySelector('.uz-title').textContent='Drop your photo here';
  document.querySelector('.uz-sub').style.display='block';
  document.getElementById('loading').innerHTML = '<div class="spinner"></div><p>Analysing your meal...</p>';
}
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML_PAGE
