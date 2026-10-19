import base64
import io

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from _kifu_core import detect_board, detect_board_debug, reverse

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


@app.post("/api/process")
async def process(file: UploadFile = File(...)):
    if file.content_type not in ("image/jpeg", "image/png"):
        raise HTTPException(400, "only image/jpeg or image/png are supported")
    raw = await file.read()
    try:
        img = Image.open(io.BytesIO(raw)).convert("RGBA")
    except Exception as e:
        raise HTTPException(400, f"cannot decode image: {e}")
    width, height = img.size
    try:
        reversed_rgba = reverse(img.tobytes())
    except Exception as e:
        raise HTTPException(500, f"core error: {e}")
    out_img = Image.frombytes("RGBA", (width, height), reversed_rgba)
    out_buf = io.BytesIO()
    out_img.save(out_buf, format="PNG")
    orig_buf = io.BytesIO()
    img.save(orig_buf, format="PNG")
    return {
        "width": width,
        "height": height,
        "original_png_b64": base64.b64encode(orig_buf.getvalue()).decode("ascii"),
        "processed_png_b64": base64.b64encode(out_buf.getvalue()).decode("ascii"),
    }


@app.post("/api/detect-board")
async def detect_board_endpoint(file: UploadFile = File(...)):
    if file.content_type not in ("image/jpeg", "image/png"):
        raise HTTPException(400, "only image/jpeg or image/png are supported")
    raw = await file.read()
    if not raw:
        raise HTTPException(400, "empty file")
    try:
        img = Image.open(io.BytesIO(raw)).convert("RGBA")
    except Exception as e:
        raise HTTPException(400, f"cannot decode image: {e}")
    width, height = img.size
    rgba = img.tobytes()
    try:
        r = detect_board(rgba, width, height)
    except Exception as e:
        raise HTTPException(500, f"core error: {e}")
    marked_img = Image.frombytes("RGBA", (width, height), r["marked_rgba"])
    marked_buf = io.BytesIO()
    marked_img.save(marked_buf, format="PNG")
    orig_buf = io.BytesIO()
    img.save(orig_buf, format="PNG")
    corners = [[int(x), int(y)] for x, y in r["corners"]]
    return {
        "found": bool(r["found"]),
        "width": width,
        "height": height,
        "original_png_b64": base64.b64encode(orig_buf.getvalue()).decode("ascii"),
        "marked_png_b64": base64.b64encode(marked_buf.getvalue()).decode("ascii"),
        "corners": corners,
    }


@app.post("/api/debug-detect")
async def debug_detect_endpoint(file: UploadFile = File(...)):
    if file.content_type not in ("image/jpeg", "image/png"):
        raise HTTPException(400, "only image/jpeg or image/png are supported")
    raw = await file.read()
    if not raw:
        raise HTTPException(400, "empty file")
    try:
        img = Image.open(io.BytesIO(raw)).convert("RGBA")
    except Exception as e:
        raise HTTPException(400, f"cannot decode image: {e}")
    width, height = img.size
    rgba = img.tobytes()
    try:
        r = detect_board_debug(rgba, width, height)
    except Exception as e:
        raise HTTPException(500, f"core error: {e}")

    images = []
    for img_data in r["images"]:
        pil = Image.frombytes("RGBA", (img_data["width"], img_data["height"]), img_data["rgba"])
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        images.append({
            "name": img_data["name"],
            "width": img_data["width"],
            "height": img_data["height"],
            "png_b64": base64.b64encode(buf.getvalue()).decode("ascii"),
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

    return {
        "found": bool(r["found"]),
        "note": str(r["note"]),
        "img_area": float(r["img_area"]),
        "width": width,
        "height": height,
        "corners": [[int(x), int(y)] for x, y in r["corners"]],
        "images": images,
        "top_contours": contours,
    }