import base64
import io

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from _kifu_core import reverse  # pybind11-модуль

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
    if not raw:
        raise HTTPException(400, "empty file")

    try:
        img = Image.open(io.BytesIO(raw)).convert("RGBA")
    except Exception as e:
        raise HTTPException(400, f"cannot decode image: {e}")

    width, height = img.size
    rgba = img.tobytes()  # bytes, 4 на пиксель

    try:
        reversed_rgba = reverse(rgba)
    except Exception as e:
        raise HTTPException(500, f"core error: {e}")

    out_img = Image.frombytes("RGBA", (width, height), reversed_rgba)
    out_buf = io.BytesIO()
    out_img.save(out_buf, format="PNG")
    out_bytes = out_buf.getvalue()

    # оригинал тоже отдадим как PNG, чтобы фронт показывал без "мигания" разных форматов
    orig_buf = io.BytesIO()
    img.save(orig_buf, format="PNG")
    orig_bytes = orig_buf.getvalue()

    return {
        "width": width,
        "height": height,
        "original_png_b64": base64.b64encode(orig_bytes).decode("ascii"),
        "processed_png_b64": base64.b64encode(out_bytes).decode("ascii"),
    }