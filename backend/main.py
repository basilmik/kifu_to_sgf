import base64
import io

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from _kifu_core import detect_board, detect_board_debug, reverse

# Максимальная длинная сторона входного изображения.
# Мобилка шлёт 4000-12000 px, но для детекции доски хватает 2000.
MAX_SIDE = 2000

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    return {"status": "ok"}


def _load_rgba(raw: bytes) -> tuple[Image.Image, float]:
    """
    Декодирует фото и ресайзит до MAX_SIDE по длинной стороне.
    Возвращает (изображение RGBA, scale), где scale = исходный_размер / новый.
    Если ресайз не нужен, scale = 1.0.
    """
    try:
        img = Image.open(io.BytesIO(raw)).convert("RGBA")
    except Exception as e:
        raise HTTPException(400, f"cannot decode image: {e}")

    w, h = img.size
    m = max(w, h)
    if m <= MAX_SIDE:
        return img, 1.0

    scale = m / MAX_SIDE
    new_w = int(w / scale)
    new_h = int(h / scale)
    img = img.resize((new_w, new_h), Image.LANCZOS)
    return img, scale


def _png_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


@app.post("/api/process")
async def process(file: UploadFile = File(...)):
    if file.content_type not in ("image/jpeg", "image/png"):
        raise HTTPException(400, "only image/jpeg or image/png are supported")

    raw = await file.read()
    img, _ = _load_rgba(raw)
    width, height = img.size

    try:
        reversed_rgba = reverse(img.tobytes())
    except Exception as e:
        raise HTTPException(500, f"core error: {e}")

    out_img = Image.frombytes("RGBA", (width, height), reversed_rgba)

    return {
        "width": width,
        "height": height,
        "original_png_b64": _png_b64(img),
        "processed_png_b64": _png_b64(out_img),
    }


@app.post("/api/detect-board")
async def detect_board_endpoint(file: UploadFile = File(...)):
    if file.content_type not in ("image/jpeg", "image/png"):
        raise HTTPException(400, "only image/jpeg or image/png are supported")

    raw = await file.read()
    if not raw:
        raise HTTPException(400, "empty file")

    img, scale = _load_rgba(raw)
    width, height = img.size

    try:
        r = detect_board(img.tobytes(), width, height)
    except Exception as e:
        raise HTTPException(500, f"core error: {e}")

    marked_img = Image.frombytes("RGBA", (width, height), r["marked_rgba"])

    # Координаты углов пересчитываем обратно в исходный масштаб,
    # чтобы они были в пикселях оригинального фото.
    corners_orig = [[int(x * scale), int(y * scale)] for x, y in r["corners"]]

    return {
        "found": bool(r["found"]),
        "width": width,
        "height": height,
        "original_png_b64": _png_b64(img),
        "marked_png_b64": _png_b64(marked_img),
        "corners": corners_orig,
    }


@app.post("/api/debug-detect")
async def debug_detect_endpoint(file: UploadFile = File(...)):
    if file.content_type not in ("image/jpeg", "image/png"):
        raise HTTPException(400, "only image/jpeg or image/png are supported")

    raw = await file.read()
    if not raw:
        raise HTTPException(400, "empty file")

    img, scale = _load_rgba(raw)
    width, height = img.size

    try:
        r = detect_board_debug(img.tobytes(), width, height)
    except Exception as e:
        raise HTTPException(500, f"core error: {e}")

    images = []
    for img_data in r["images"]:
        pil = Image.frombytes(
            "RGBA", (img_data["width"], img_data["height"]), img_data["rgba"]
        )
        images.append({
            "name": img_data["name"],
            "width": img_data["width"],
            "height": img_data["height"],
            "png_b64": _png_b64(pil),
        })

    contours = []
    for ci in r["top_contours"]:
        contours.append({
            "area": float(ci["area"]),
            "perimeter": float(ci["perimeter"]),
            "approx_points": int(ci["approx_points"]),
            "reasonable": bool(ci["reasonable"]),
            "points": [[int(x), int(y)] for x, y in ci["points"]],
        })

    corners_orig = [[int(x * scale), int(y * scale)] for x, y in r["corners"]]

    return {
        "found": bool(r["found"]),
        "note": str(r["note"]),
        "img_area": float(r["img_area"]),
        "width": width,
        "height": height,
        "corners": corners_orig,
        "images": images,
        "top_contours": contours,
    }