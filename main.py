import os
import csv
import io
import asyncio
import httpx
import anthropic
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel
from typing import Optional

load_dotenv(Path(__file__).parent / ".env", override=True)

app = FastAPI(title="Amazon Rank Tracker")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

SERPAPI_KEY = os.getenv("SERPAPI_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

SERPAPI_BASE = "https://serpapi.com/search"
MAX_PAGES = 3
PAGE_SIZE = 16  # Amazon typically shows 16-24 results per page


class AsinRequest(BaseModel):
    asin: str


class SuggestRequest(BaseModel):
    asin: str
    product_title: str


class RankingRequest(BaseModel):
    asin: str
    terms: list[str]
    marketplace: Optional[str] = "amazon.com"


def check_api_keys():
    if not SERPAPI_KEY:
        raise HTTPException(status_code=500, detail="SERPAPI_KEY not configured. Add it to your .env file.")
    if not ANTHROPIC_API_KEY:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not configured. Add it to your .env file.")


@app.get("/")
async def root():
    return FileResponse("static/index.html")


@app.post("/api/product-info")
async def get_product_info(req: AsinRequest):
    if not SERPAPI_KEY:
        raise HTTPException(status_code=500, detail="SERPAPI_KEY not configured.")

    asin = req.asin.strip().upper()
    if not asin:
        raise HTTPException(status_code=400, detail="ASIN cannot be empty.")

    params = {
        "engine": "amazon_product",
        "asin": asin,
        "amazon_domain": "amazon.com",
        "api_key": SERPAPI_KEY,
    }

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(SERPAPI_BASE, params=params)

    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail=f"SerpAPI error: {resp.text[:200]}")

    data = resp.json()

    if "error" in data:
        raise HTTPException(status_code=400, detail=data["error"])

    product = data.get("product_results", {})
    if not product:
        raise HTTPException(status_code=404, detail=f"No product found for ASIN: {asin}")

    return {
        "asin": asin,
        "title": product.get("title", "Unknown Product"),
        "brand": product.get("brand", ""),
        "image": product.get("media", [{}])[0].get("link", "") if product.get("media") else product.get("image", ""),
        "rating": product.get("rating", None),
        "ratings_total": product.get("ratings_total", None),
        "price": product.get("price", {}).get("current", {}).get("value", None) if isinstance(product.get("price"), dict) else None,
    }


@app.post("/api/suggest-terms")
async def suggest_terms(req: SuggestRequest):
    if not ANTHROPIC_API_KEY:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not configured.")

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    prompt = f"""You are an Amazon SEO expert. Given this Amazon product, suggest 10 highly relevant search terms that real customers would type into Amazon's search bar to find this product.

Product ASIN: {req.asin}
Product Title: {req.product_title}

Rules:
- Return ONLY the search terms, one per line, no numbering, no explanation
- Include a mix of: broad terms, specific terms, long-tail phrases, use-case terms
- Think about what a real buyer would search for
- Terms should be 1-6 words each
- Do not include the ASIN itself"""

    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = message.content[0].text.strip()
    terms = [line.strip() for line in raw.splitlines() if line.strip()]
    # Remove any accidental numbering (1. 2. etc)
    cleaned = []
    for t in terms:
        if t and len(t) > 1:
            # Strip leading "1." or "1)" patterns
            if len(t) > 2 and t[0].isdigit() and t[1] in ".):":
                t = t[2:].strip()
            elif len(t) > 3 and t[0].isdigit() and t[1].isdigit() and t[2] in ".):":
                t = t[3:].strip()
            if t:
                cleaned.append(t)

    return {"terms": cleaned[:10]}


async def search_term_ranking(client: httpx.AsyncClient, asin: str, term: str, marketplace: str) -> dict:
    """Search Amazon for a keyword and find the ASIN's position."""
    asin = asin.upper()
    position_overall = None
    found_page = None

    for page_num in range(1, MAX_PAGES + 1):
        params = {
            "engine": "amazon",
            "k": term,
            "amazon_domain": marketplace,
            "page": page_num,
            "api_key": SERPAPI_KEY,
        }

        try:
            resp = await client.get(SERPAPI_BASE, params=params, timeout=30)
            if resp.status_code != 200:
                break

            data = resp.json()
            if "error" in data:
                break

            organic = data.get("organic_results", [])
            if not organic:
                break

            for item in organic:
                item_asin = (item.get("asin") or "").upper()
                if item_asin == asin:
                    pos_in_page = item.get("position", organic.index(item) + 1)
                    position_overall = (page_num - 1) * PAGE_SIZE + pos_in_page
                    found_page = page_num
                    break

            if position_overall is not None:
                break

        except Exception:
            break

    return {
        "term": term,
        "position": position_overall,
        "page": found_page,
        "found": position_overall is not None,
    }


@app.post("/api/check-rankings")
async def check_rankings(req: RankingRequest):
    if not SERPAPI_KEY:
        raise HTTPException(status_code=500, detail="SERPAPI_KEY not configured.")
    if not req.terms:
        raise HTTPException(status_code=400, detail="No search terms provided.")

    asin = req.asin.strip().upper()
    marketplace = req.marketplace or "amazon.com"

    async with httpx.AsyncClient() as client:
        tasks = [
            search_term_ranking(client, asin, term.strip(), marketplace)
            for term in req.terms
            if term.strip()
        ]
        results = await asyncio.gather(*tasks)

    return {"asin": asin, "results": list(results)}


@app.get("/api/template.csv")
async def download_csv_template():
    rows = [
        ["ASIN", "Marketplace", "Search Term 1", "Search Term 2", "Search Term 3",
         "Search Term 4", "Search Term 5", "Search Term 6", "Search Term 7", "Search Term 8"],
        ["B08N5WRWNW", "amazon.com", "echo dot", "alexa speaker", "smart speaker",
         "echo dot 4th gen", "amazon echo", "voice assistant speaker", "", ""],
        ["B07XJ8C8F7", "amazon.com", "fire tv stick", "streaming device", "hdmi streaming",
         "4k fire stick", "amazon fire tv", "", "", ""],
    ]
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerows(rows)
    return Response(
        content=buf.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=asin_rank_template.csv"},
    )


def parse_upload(content: bytes, filename: str) -> list[dict]:
    """Parse CSV or XLSX upload. Returns list of {asin, marketplace, terms}."""
    rows = []

    if filename.lower().endswith(".xlsx"):
        from openpyxl import load_workbook
        wb = load_workbook(io.BytesIO(content), read_only=True, data_only=True)
        ws = wb.worksheets[0]
        raw_rows = [[str(cell.value).strip() if cell.value is not None else "" for cell in row]
                    for row in ws.iter_rows()]
        wb.close()
    else:
        text = content.decode("utf-8-sig", errors="replace")
        raw_rows = [row for row in csv.reader(text.splitlines())]

    if not raw_rows:
        return rows

    # Skip header row
    data_rows = raw_rows[1:]

    for row in data_rows:
        if not row:
            continue
        asin = row[0].strip().upper() if len(row) > 0 else ""
        if not asin or asin in ("ASIN", ""):
            continue
        marketplace = row[1].strip() if len(row) > 1 and row[1].strip() else "amazon.com"
        terms = [t.strip() for t in row[2:] if len(row) > 2 and t.strip()]
        if asin and terms:
            rows.append({"asin": asin, "marketplace": marketplace, "terms": terms})

    return rows


@app.post("/api/bulk-check")
async def bulk_check(file: UploadFile = File(...)):
    if not SERPAPI_KEY:
        raise HTTPException(status_code=500, detail="SERPAPI_KEY not configured.")

    filename = file.filename or ""
    if not (filename.lower().endswith(".csv") or filename.lower().endswith(".xlsx")):
        raise HTTPException(status_code=400, detail="Only .csv and .xlsx files are supported.")

    content = await file.read()
    try:
        entries = parse_upload(content, filename)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse file: {e}")

    if not entries:
        raise HTTPException(status_code=400, detail="No valid ASIN + search term rows found. Check the file format.")

    results = []
    async with httpx.AsyncClient() as client:
        for entry in entries:
            asin = entry["asin"]
            marketplace = entry["marketplace"]
            tasks = [
                search_term_ranking(client, asin, term, marketplace)
                for term in entry["terms"]
            ]
            rankings = await asyncio.gather(*tasks)
            results.append({
                "asin": asin,
                "marketplace": marketplace,
                "rankings": list(rankings),
            })

    return {"results": results}


app.mount("/static", StaticFiles(directory="static"), name="static")
