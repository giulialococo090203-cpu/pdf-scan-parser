import os
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import fitz
from PIL import Image
from paddleocr import PaddleOCR
import io
import numpy as np
import re

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
        "https://magazzino-pro.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ocr = None

def get_ocr():
    global ocr
    if ocr is None:
        ocr = PaddleOCR(
            use_doc_orientation_classify=True,
            use_doc_unwarping=True,
            use_textline_orientation=True,
            lang="en",
        )
    return ocr


def render_first_page_to_image(file_bytes: bytes):
    try:
        pdf = fitz.open(stream=file_bytes, filetype="pdf")
    except Exception as e:
        raise ValueError(f"PDF non valido: {e}")

    if len(pdf) == 0:
        raise ValueError("PDF senza pagine.")

    page = pdf[0]
    matrix = fitz.Matrix(1.5, 1.5)
    pix = page.get_pixmap(matrix=matrix, alpha=False)
    image = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
    return image


def crop_invoice_table_area(image: Image.Image):
    w, h = image.size
    left = int(w * 0.00)
    top = int(h * 0.35)
    right = int(w * 0.98)
    bottom = int(h * 0.76)
    cropped = image.crop((left, top, right, bottom))
    return cropped, {
        "left": left,
        "top": top,
        "right": right,
        "bottom": bottom,
        "width": cropped.size[0],
        "height": cropped.size[1],
    }


def run_ocr_on_image(image: Image.Image):
    image_np = np.array(image)
    engine = get_ocr()
    return engine.predict(image_np)


def simplify_paddle_result(results):
    simplified = []

    for res in results:
        payload = getattr(res, "json", res)

        if isinstance(payload, dict) and "res" in payload:
            payload = payload["res"]

        if not isinstance(payload, dict):
            simplified.append({"raw": str(payload)})
            continue

        rec_texts = payload.get("rec_texts", []) or []
        rec_scores = payload.get("rec_scores", []) or []
        dt_polys = payload.get("dt_polys", []) or []

        items = []
        for i, text in enumerate(rec_texts):
            box = dt_polys[i] if i < len(dt_polys) else None

            x_center = y_center = x_min = x_max = y_min = y_max = None
            if box and isinstance(box, list) and len(box) >= 4:
                pts = [p for p in box if isinstance(p, list) and len(p) >= 2]
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                if xs and ys:
                    x_center = sum(xs) / len(xs)
                    y_center = sum(ys) / len(ys)
                    x_min, x_max = min(xs), max(xs)
                    y_min, y_max = min(ys), max(ys)

            items.append({
                "text": str(text).strip(),
                "score": rec_scores[i] if i < len(rec_scores) else None,
                "box": box,
                "x": x_center,
                "y": y_center,
                "x_min": x_min,
                "x_max": x_max,
                "y_min": y_min,
                "y_max": y_max,
            })

        simplified.append({"items_found": len(items), "items": items})

    return simplified


def flatten_items(simplified):
    flat_items = []
    for block in simplified:
        for item in block.get("items", []):
            flat_items.append(item)
    return sorted(flat_items, key=lambda x: ((x.get("y") or 0), (x.get("x") or 0)))


def normalize_text(value: str) -> str:
    return " ".join(str(value or "").strip().lower().split())


def parse_number(value: str):
    text = str(value or "").strip().replace(".", "").replace(",", ".")
    text = re.sub(r"[^0-9.\-]", "", text)
    try:
        return float(text)
    except Exception:
        return None


def looks_like_code(text: str) -> bool:
    t = str(text or "").strip().upper()
    return bool(re.fullmatch(r"[A-Z][0-9A-Z]{4,}", t))


def looks_like_number(text: str) -> bool:
    return parse_number(text) is not None


def build_matrix(extracted_rows):
    return [
        ["Codice", "Descrizione", "Quantità", "UM", "Prezzo", "Marca", "Categoria", "Posizione"],
        *[
            [
                row.get("code", ""),
                row.get("description", ""),
                row.get("quantity", 0) or 0,
                row.get("unit", "PZ") or "PZ",
                row.get("price", 0) or 0,
                "",
                "",
                "A1-01",
            ]
            for row in extracted_rows
        ]
    ]


def find_header_items(items):
    header_map = {}
    targets = {
        "code": ["codice"],
        "description": ["descrizione"],
        "unit": ["um"],
        "quantity": ["quantità", "quantita", "qta", "qtà"],
        "price": ["prezzo u.", "prezzo u", "prezzo"],
        "total": ["totale"],
    }

    for item in items:
        t = normalize_text(item["text"])
        for key, variants in targets.items():
            if t in variants and key not in header_map:
                header_map[key] = item

    return header_map


def find_stop_y(items, header_y):
    candidates = []
    stop_words = [
        "totale merce",
        "trasporto",
        "imballo",
        "spese",
        "scadenze",
        "annotazioni",
        "totale documento",
        "totale fattura",
        "imponibile",
        "aliquota",
        "imposta",
        "vettore",
    ]
    for item in items:
        y = item.get("y")
        if y is None or y <= header_y:
            continue
        t = normalize_text(item["text"])
        if any(sw in t for sw in stop_words):
            candidates.append(y)
    if not candidates:
        return header_y + 900
    return min(candidates) - 10


def build_column_boundaries(headers):
    quantity_x = headers.get("quantity", {}).get("x")
    if quantity_x is None:
        quantity_x = (headers["unit"]["x"] + headers["price"]["x"]) / 2

    return {
        "code": headers["code"]["x"],
        "description": headers["description"]["x"],
        "unit": headers["unit"]["x"],
        "quantity": quantity_x,
        "price": headers["price"]["x"],
        "total": headers["total"]["x"],
    }


def extract_rows_full_page(items):
    valid_items = [i for i in items if i.get("text")]
    headers = find_header_items(valid_items)

    required = ["code", "description", "unit", "price", "total"]
    if not all(k in headers for k in required):
        return [], {"reason": "headers_not_found", "headers": headers}

    header_y_values = [headers[k]["y"] for k in headers if headers[k].get("y") is not None]
    header_y = sum(header_y_values) / len(header_y_values)
    stop_y = find_stop_y(valid_items, header_y)
    col_x = build_column_boundaries(headers)

    article_zone = [
        i for i in valid_items
        if i.get("y") is not None and (header_y + 8) < i["y"] < stop_y
    ]

    code_anchors = []
    for item in article_zone:
        x = item.get("x")
        text = item.get("text", "")
        if x is None:
            continue
        if x < col_x["description"] - 20 and looks_like_code(text):
            code_anchors.append(item)

    code_anchors = sorted(code_anchors, key=lambda i: i["y"] or 0)
    extracted = []

    for idx, anchor in enumerate(code_anchors):
        ay = anchor["y"] or 0
        next_anchor_y = code_anchors[idx + 1]["y"] if idx + 1 < len(code_anchors) else None

        start_y = ay - 10
        end_y = min(next_anchor_y - 10, ay + 75) if next_anchor_y else min(stop_y, ay + 75)

        band_items = [
            i for i in article_zone
            if i.get("y") is not None and start_y <= i["y"] <= end_y
        ]

        description_parts = []
        unit = ""
        quantity_raw = ""
        price_raw = ""
        total_candidates = []

        for item in band_items:
            text = item.get("text", "").strip()
            x = item.get("x")
            if not text or x is None:
                continue
            if item is anchor:
                continue

            if col_x["description"] - 40 <= x < col_x["unit"] - 15:
                description_parts.append((item["y"] or 0, text))
            elif col_x["unit"] - 15 <= x < col_x["quantity"] - 15:
                if not unit:
                    unit = text
            elif col_x["quantity"] - 20 <= x < col_x["price"] - 15:
                if looks_like_number(text) and not quantity_raw:
                    quantity_raw = text
            elif col_x["price"] - 15 <= x < col_x["total"] - 15:
                if looks_like_number(text) and not price_raw:
                    price_raw = text
            elif x >= col_x["total"] - 15:
                if looks_like_number(text):
                    total_candidates.append((x, text))

        description = " ".join(text for _, text in sorted(description_parts, key=lambda t: t[0])).strip()
        description = re.sub(r"\s+", " ", description)

        total_raw = ""
        if total_candidates:
            numeric_totals = []
            for x, txt in total_candidates:
                num = parse_number(txt)
                if num is None:
                    continue
                if num > 1:
                    numeric_totals.append((x, txt, num))
            if numeric_totals:
                numeric_totals = sorted(numeric_totals, key=lambda t: t[0])
                total_raw = numeric_totals[0][1]

        row = {
            "code": anchor["text"].strip(),
            "description": description,
            "unit": unit,
            "quantity_raw": quantity_raw,
            "price_raw": price_raw,
            "total_raw": total_raw,
            "quantity": parse_number(quantity_raw),
            "price": parse_number(price_raw),
            "total": parse_number(total_raw),
        }

        if (
            row["code"]
            and row["description"]
            and row["quantity"] not in (None, 0)
            and row["price"] is not None
        ):
            extracted.append(row)

    return extracted, {
        "strategy": "full_page",
        "header_y": header_y,
        "stop_y": stop_y,
        "col_x": col_x,
        "code_anchors": [{"text": a["text"], "x": a["x"], "y": a["y"]} for a in code_anchors],
        "itemsFound": len(valid_items),
    }


def extract_rows_cropped(items):
    valid_items = [i for i in items if i.get("text")]
    if not valid_items:
        return [], {"strategy": "cropped", "itemsFound": 0}

    col_description = 190
    col_unit = 585
    col_quantity = 700
    col_price = 810
    col_total = 1065

    code_anchors = []
    for item in valid_items:
        x = item.get("x")
        text = item.get("text", "")
        if x is None:
            continue
        if x < 120 and looks_like_code(text):
            code_anchors.append(item)

    code_anchors = sorted(code_anchors, key=lambda i: i["y"] or 0)
    if not code_anchors:
        return [], {
            "strategy": "cropped",
            "reason": "no_code_anchors",
            "itemsFound": len(valid_items),
        }

    extracted = []

    for idx, anchor in enumerate(code_anchors):
        ay = anchor["y"] or 0
        next_anchor_y = code_anchors[idx + 1]["y"] if idx + 1 < len(code_anchors) else None

        start_y = ay - 8
        end_y = min(next_anchor_y - 10, ay + 75) if next_anchor_y else ay + 75

        band_items = [
            i for i in valid_items
            if i.get("y") is not None and start_y <= i["y"] <= end_y
        ]

        description_parts = []
        unit = ""
        quantity_raw = ""
        price_raw = ""
        total_candidates = []

        for item in band_items:
            text = item.get("text", "").strip()
            x = item.get("x")
            if not text or x is None:
                continue
            if item is anchor:
                continue

            if col_description - 40 <= x < col_unit - 15:
                description_parts.append((item["y"] or 0, text))
            elif col_unit - 15 <= x < col_quantity - 15:
                if not unit:
                    unit = text
            elif col_quantity - 20 <= x < col_price - 15:
                if looks_like_number(text) and not quantity_raw:
                    quantity_raw = text
            elif col_price - 15 <= x < col_total - 15:
                if looks_like_number(text) and not price_raw:
                    price_raw = text
            elif x >= col_total - 15:
                if looks_like_number(text):
                    total_candidates.append((x, text))

        description = " ".join(text for _, text in sorted(description_parts, key=lambda t: t[0])).strip()
        description = re.sub(r"\s+", " ", description)

        total_raw = ""
        if total_candidates:
            numeric_totals = []
            for x, txt in total_candidates:
                num = parse_number(txt)
                if num is None:
                    continue
                if num > 1:
                    numeric_totals.append((x, txt, num))
            if numeric_totals:
                numeric_totals = sorted(numeric_totals, key=lambda t: t[0])
                total_raw = numeric_totals[0][1]

        row = {
            "code": anchor["text"].strip(),
            "description": description,
            "unit": unit,
            "quantity_raw": quantity_raw,
            "price_raw": price_raw,
            "total_raw": total_raw,
            "quantity": parse_number(quantity_raw),
            "price": parse_number(price_raw),
            "total": parse_number(total_raw),
        }

        if (
            row["code"]
            and row["description"]
            and row["quantity"] not in (None, 0)
            and row["price"] is not None
        ):
            extracted.append(row)

    return extracted, {
        "strategy": "cropped",
        "itemsFound": len(valid_items),
        "code_anchors": [{"text": a["text"], "x": a["x"], "y": a["y"]} for a in code_anchors],
    }


def parse_hybrid(image: Image.Image):
    cropped, crop_meta = crop_invoice_table_area(image)
    cropped_results = run_ocr_on_image(cropped)
    cropped_items = flatten_items(simplify_paddle_result(cropped_results))
    cropped_rows, cropped_debug = extract_rows_cropped(cropped_items)

    if cropped_rows:
        return cropped_rows, {
            "mode": "cropped-fast",
            "crop": crop_meta,
            "debug": cropped_debug,
        }

    full_results = run_ocr_on_image(image)
    full_items = flatten_items(simplify_paddle_result(full_results))
    full_rows, full_debug = extract_rows_full_page(full_items)

    if full_rows:
        return full_rows, {
            "mode": "full-page-fallback",
            "crop": crop_meta,
            "cropped_debug": cropped_debug,
            "debug": full_debug,
        }

    return [], {
        "mode": "failed",
        "crop": crop_meta,
        "cropped_debug": cropped_debug,
        "debug": full_debug if "full_debug" in locals() else {},
        "cropped_items_found": len(cropped_items),
        "full_items_found": len(full_items) if "full_items" in locals() else 0,
    }


@app.api_route("/", methods=["GET", "HEAD"])
def root():
    return {
        "ok": True,
        "service": "pdf-scan-parser",
        "step": "hybrid-ready"
    }


@app.post("/debug-scan-invoice")
async def debug_scan_invoice(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="Nessun file ricevuto.")

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="File vuoto.")

    try:
        image = render_first_page_to_image(file_bytes)
        extracted_rows, debug = parse_hybrid(image)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {
        "ok": True,
        "fileName": file.filename,
        "extractedRows": extracted_rows,
        "debug": debug,
    }


@app.post("/parse-scan-invoice")
async def parse_scan_invoice(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="Nessun file ricevuto.")

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="File vuoto.")

    try:
        image = render_first_page_to_image(file_bytes)
        extracted_rows, debug = parse_hybrid(image)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    if not extracted_rows:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Nessuna riga articolo riconosciuta nella scansione.",
                "debug": debug,
            }
        )

    matrix = build_matrix(extracted_rows)

    return {
        "ok": True,
        "mode": "scan-ocr",
        "scanDetected": False,
        "fileName": file.filename,
        "extractedRows": extracted_rows,
        "matrix": matrix,
        "debug": debug,
    }
