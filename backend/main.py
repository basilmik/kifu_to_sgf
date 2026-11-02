import base64
import io

import binascii

from pydantic import BaseModel
from fastapi import FastAPI, File, HTTPException, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from ocr_baseline import compare_ocr
from PIL import Image

from _kifu_core import detect_board, detect_board_debug, reverse, rectify_grid

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


@app.post("/api/grid")
async def grid_endpoint(
    file: UploadFile = File(...),
    manual_corners: str | None = Form(None),
    grid_size: int = Form(19),
):
    if file.content_type not in ("image/jpeg", "image/png"):
        raise HTTPException(
            400,
            "only JPEG and PNG are supported"
        )

    if grid_size not in (9, 13, 19):
        raise HTTPException(
            422,
            "grid_size must be 9, 13 or 19"
        )

    raw = await file.read()

    if not raw:
        raise HTTPException(400, "empty file")

    img, scale = _load_rgba(raw)

    width, height = img.size

    # Углы пользователь передаёт в координатах
    # оригинального изображения (после EXIF-поворота).
    corners_scaled = []

    if manual_corners:
        try:
            points = json.loads(manual_corners)

            if not isinstance(points, list) or len(points) != 4:
                raise ValueError("exactly 4 corners required")

            for point in points:
                if not isinstance(point, list) or len(point) != 2:
                    raise ValueError("each corner must be [x, y]")

                x = float(point[0])
                y = float(point[1])

                if not math.isfinite(x) or not math.isfinite(y):
                    raise ValueError("non-finite coordinates")

                corners_scaled.append((
                    x / scale,
                    y / scale
                ))

        except (ValueError, TypeError) as e:
            raise HTTPException(
                422,
                f"invalid manual_corners: {e}"
            )

    try:
        result = rectify_grid(
            img.tobytes(),
            width,
            height,
            corners_scaled,
            grid_size,
            1400
        )

    except ValueError as e:
        raise HTTPException(422, str(e))

    except Exception as e:
        raise HTTPException(
            500,
            f"grid processing error: {e}"
        )

    # Восстанавливаем выпрямленное изображение.
    rectified_png_b64 = None

    side = result["rectified_side"]

    if side > 0:
        rectified = Image.frombytes(
            "RGBA",
            (side, side),
            result["rectified_rgba"]
        )

        rectified_png_b64 = _png_b64(rectified)

    # Координаты сетки C++ возвращает в масштабе
    # уменьшенной фотографии.
    # Переводим их в исходный масштаб.
    grid_points_original = [
        [round(x * scale, 2), round(y * scale, 2)]
        for x, y in result["grid_points_image"]
    ]

    corners_original = [
        [round(x * scale, 2), round(y * scale, 2)]
        for x, y in result["corners"]
    ]

    return {
        "found": result["found"],
        "note": result["note"],
        "grid_size": result["grid_size"],

        "original_width": round(width * scale),
        "original_height": round(height * scale),

        "rectified_side": side,
        "rectified_png_b64": rectified_png_b64,

        "corners_original": corners_original,
        "grid_points_original": grid_points_original,

        "x_lines": result["x_lines"],
        "y_lines": result["y_lines"],

        "x_quality": result["x_quality"],
        "y_quality": result["y_quality"]
    }


class RecognizeRequest(BaseModel):
    rectified_png_b64: str

    grid_size: int = 19

    x_lines: list[float]
    y_lines: list[float]

    max_move: int = 500

@app.post("/api/recognize-sgf")
async def recognize_sgf_endpoint(req: RecognizeRequest):

    if req.grid_size not in (9, 13, 19):
        raise HTTPException(
            422,
            "unsupported grid size"
        )

    if (
        len(req.x_lines) != req.grid_size
        or len(req.y_lines) != req.grid_size
    ):
        raise HTTPException(
            422,
            "invalid grid line count"
        )

    if not 1 <= req.max_move <= 2000:
        raise HTTPException(
            422,
            "invalid max_move"
        )

    # Ограничиваем размер входного base64.
    if len(req.rectified_png_b64) > 20_000_000:
        raise HTTPException(
            413,
            "image payload too large"
        )

    try:
        raw = base64.b64decode(
            req.rectified_png_b64,
            validate=True
        )

        with Image.open(io.BytesIO(raw)) as source:
            if source.format != "PNG":
                raise ValueError("expected PNG image")

            if max(source.size) > 4096:
                raise ValueError("image is too large")

            image = source.convert("RGB")

    except (
        ValueError,
        binascii.Error,
        OSError,
    ) as e:
        raise HTTPException(
            400,
            f"invalid image: {e}"
        )

    try:
        # OCR может выполняться заметное время.
        # Выносим синхронную работу из event loop.
        result = await run_in_threadpool(
            recognize_numbers,
            image,
            req.x_lines,
            req.y_lines,
            req.max_move
        )

        return result

    except ValueError as e:
        raise HTTPException(422, str(e))