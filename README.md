# Meal Scanner 🍽️🔬

Upload a photo of a meal → get back **ingredients**, **calories**, and **macros**, powered by Google Gemini Flash.

## Quick start

```bash
# 1. install
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. add your API key
cp .env.example .env
# edit .env → GEMINI_API_KEY=your_key_here

# 3. run
python main.py
# → http://localhost:8000
```

## API

**POST** `/analyze`

Send a JPEG/PNG/WebP file as `multipart/form-data` under the field name `file`.

```bash
curl -X POST http://localhost:8000/analyze \
  -F "file=@foto-do-prato.jpg"
```

**Response:**

```json
{
  "dish_name": "Bacalhau à Brás",
  "ingredients": [
    {"name": "Bacalhau desfiado", "estimated_amount": "150g"},
    {"name": "Batata palha", "estimated_amount": "120g"},
    {"name": "Ovo", "estimated_amount": "2 unidades"},
    {"name": "Cebola", "estimated_amount": "60g"},
    {"name": "Azeite", "estimated_amount": "15ml"}
  ],
  "total_calories_kcal": 485,
  "protein_g": 32,
  "carbs_g": 38,
  "fat_g": 22,
  "notes": "Prato típico português. A batata palha aumenta o teor de hidratos."
}
```

## CLI

```bash
python main.py foto-do-prato.jpg
```

## Pricing

Gemini 2.0 Flash: ~$0.075/1M input tokens, $0.30/1M output.  
A typical photo analysis costs **<$0.001** per call.

## Stack

- **FastAPI** — framework
- **google-genai** — Gemini SDK
- **Pillow** — image validation & resize
- **python-multipart** — file upload
