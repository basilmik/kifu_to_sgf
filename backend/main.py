import base64
import io

import binascii
import json
import math

from pydantic import BaseModel
from fastapi import FastAPI, File, HTTPException, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from ocr_baseline import recognize_numbers, compare_ocr
from compare_digits import compare_digit_models, _mnist_digit
from PIL import Image

from _kifu_core import detect_board, detect_board_debug, reverse, rectify_grid

# Максимальная длинная сторона входного изображения.
# Мобилка шлёт 4000-12000 px, но для детекции доски хватает 2000.
MAX_SIDE = 4000

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


def _np_png_b64(arr) -> str:
    import cv2
    ok, png = cv2.imencode(".png", arr)
    if not ok:
        return ""
    return base64.b64encode(png.tobytes()).decode("ascii")


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
        raise HTTPException(400, "only JPEG and PNG are supported")

    if grid_size not in (9, 13, 19):
        raise HTTPException(422, "grid_size must be 9, 13 or 19")

    raw = await file.read()

    if not raw:
        raise HTTPException(400, "empty file")

    img, scale = _load_rgba(raw)
    width, height = img.size

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

                corners_scaled.append((x / scale, y / scale))

        except (ValueError, TypeError) as e:
            raise HTTPException(422, f"invalid manual_corners: {e}")

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
        raise HTTPException(500, f"grid processing error: {e}")

    rectified_png_b64 = None
    side = result["rectified_side"]

    if side > 0:
        rectified = Image.frombytes(
            "RGBA",
            (side, side),
            result["rectified_rgba"]
        )
        rectified_png_b64 = _png_b64(rectified)

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
    compare_models: bool = False
    max_rois: int = 100


@app.post("/api/recognize-sgf")
async def recognize_sgf_endpoint(req: RecognizeRequest):

    if req.grid_size not in (9, 13, 19):
        raise HTTPException(422, "unsupported grid size")

    if (
        len(req.x_lines) != req.grid_size
        or len(req.y_lines) != req.grid_size
    ):
        raise HTTPException(422, "invalid grid line count")

    if not 1 <= req.max_move <= 2000:
        raise HTTPException(422, "invalid max_move")

    if len(req.rectified_png_b64) > 20_000_000:
        raise HTTPException(413, "image payload too large")

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

    except (ValueError, binascii.Error, OSError) as e:
        raise HTTPException(400, f"invalid image: {e}")

    try:
        if req.compare_models:
            if not 1 <= req.max_rois <= req.grid_size ** 2:
                raise HTTPException(422, "invalid max_rois")

            return await run_in_threadpool(
                compare_ocr,
                image,
                req.x_lines,
                req.y_lines,
                req.max_rois,
            )

        result = await run_in_threadpool(
            recognize_numbers,
            image,
            req.x_lines,
            req.y_lines,
            req.max_move,
        )

        return result

    except ValueError as e:
        raise HTTPException(422, str(e))


class CompareDigitsRequest(BaseModel):
    rectified_png_b64: str
    grid_size: int = 19
    x_lines: list[float]
    y_lines: list[float]
    max_rois: int = 6
    cells: list[tuple[int, int]] | None = None


@app.post("/api/compare-digit-models")
async def compare_digit_models_endpoint(req: CompareDigitsRequest):
    if req.grid_size not in (9, 13, 19):
        raise HTTPException(422, "invalid grid size")

    if (
        len(req.x_lines) != req.grid_size
        or len(req.y_lines) != req.grid_size
    ):
        raise HTTPException(422, "invalid grid lines")

    if not 1 <= req.max_rois <= 12:
        raise HTTPException(422, "invalid max_rois")

    if len(req.rectified_png_b64) > 60_000_000:
        raise HTTPException(413, "image too large")

    try:
        raw = base64.b64decode(
            req.rectified_png_b64,
            validate=True,
        )

        with Image.open(io.BytesIO(raw)) as source:
            if source.format != "PNG":
                raise ValueError("expected PNG")

            if max(source.size) > 6000:
                raise ValueError("image too large")

            image = source.convert("RGB")

    except (ValueError, binascii.Error, OSError) as e:
        raise HTTPException(400, f"invalid image: {e}")

    try:
        return await run_in_threadpool(
            compare_digit_models,
            image,
            req.x_lines,
            req.y_lines,
            req.max_rois,
            req.cells,
        )

    except ValueError as e:
        raise HTTPException(422, str(e))


class ExtractCellsRequest(BaseModel):
    rectified_png_b64: str
    grid_size: int = 19
    x_lines: list[float]
    y_lines: list[float]


def _remove_circle_mark(crop):
    """
    Ищет контур кружка/овала, обводящего ход, и убирает его
    из кропа (осветляет/затирает). Цифру внутри не трогает.

    Возвращает (cleaned_crop, debug):
        cleaned_crop — uint8 grayscale, тот же размер, что crop;
        debug — dict с полями:
            "circle_found": bool,
            "circle_png_b64": base64 PNG маски кружка (белый контур на чёрном)
            "circle_bbox": [x1, y1, x2, y2] или None
            "circle_circularity": float
            "note": str
    """
    import numpy as np
    import cv2

    if crop.size == 0:
        return crop, {
            "circle_found": False,
            "circle_png_b64": "",
            "circle_bbox": None,
            "circle_circularity": 0.0,
            "note": "empty",
        }

    h, w = crop.shape

    # 1. Инверсия + бинаризация: штрихи белые на чёрном.
    inverted = 255 - crop
    _, binary = cv2.threshold(
        inverted, 0, 255,
        cv2.THRESH_BINARY | cv2.THRESH_OTSU,
    )

    # 2. Ищем контуры.
    contours, _ = cv2.findContours(
        binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return crop, {
            "circle_found": False,
            "circle_png_b64": "",
            "circle_bbox": None,
            "circle_circularity": 0.0,
            "note": "no_contours",
        }

    # 3. Ищем лучший кандидат в кружок.
    # Кружок: большой bbox (не меньше 50% кропа), высокая
    # circularity, низкое заполнение bbox (внутри пусто/цифра).
    best = None
    best_score = 0.0

    crop_area = float(h * w)
    min_bbox_side = 0.45 * min(h, w)

    for c in contours:
        area = float(cv2.contourArea(c))
        if area <= 0:
            continue

        peri = float(cv2.arcLength(c, True))
        if peri <= 0:
            continue

        x, y, bw, bh = cv2.boundingRect(c)

        # Кружок должен быть крупным и почти квадратным в bbox.
        if bw < min_bbox_side or bh < min_bbox_side:
            continue

        aspect = max(bw, bh) / max(1.0, min(bw, bh))
        if aspect > 1.6:
            continue

        circularity = 4.0 * math.pi * area / (peri * peri)

        # Заполнение bbox: у кружка контур тонкий, extent низкий.
        bbox_area = float(bw * bh)
        extent = area / bbox_area if bbox_area > 0 else 0.0

        # Порог circularity для эллипса: >0.55. Для цифры: <0.3.
        if circularity < 0.55:
            continue

        # Кружок занимает большую часть кропа.
        if bbox_area < 0.35 * crop_area:
            continue

        # Контур тонкий — значит это обводка, а не заполненная фигура.
        if extent > 0.6:
            continue

        score = circularity + (1.0 - extent) * 0.5
        if score > best_score:
            best_score = score
            best = (c, area, peri, circularity, extent, (x, y, bw, bh))

    if best is None:
        return crop, {
            "circle_found": False,
            "circle_png_b64": "",
            "circle_bbox": None,
            "circle_circularity": 0.0,
            "note": "no_circle_candidate",
        }

    c, area, peri, circularity, extent, bbox = best

    # 4. Строим маску контура кружка с небольшой дилатацией,
    # чтобы убрать антиалиасинг.
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(mask, [c], -1, 255, thickness=1)

    # Дилатация 1 px: только чтобы затронуть края штриха.
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)

    # 5. Проверяем, что маска не покрывает слишком много
    # пикселей цифры. Цифра обычно внутри bbox и имеет
    # не слишком большую площадь. Если маска покрывает
    # больше 40% тёмных пикселей кропа — не рискуем.
    dark = binary > 0
    dark_total = int(np.count_nonzero(dark))
    if dark_total == 0:
        return crop, {
            "circle_found": False,
            "circle_png_b64": "",
            "circle_bbox": None,
            "circle_circularity": 0.0,
            "note": "no_dark_pixels",
        }

    overlap = int(np.count_nonzero(mask & binary))
    if overlap > 0.4 * dark_total:
        return crop, {
            "circle_found": False,
            "circle_png_b64": "",
            "circle_bbox": None,
            "circle_circularity": 0.0,
            "note": f"mask_too_large:{overlap}/{dark_total}",
        }

    # 6. Осветляем пиксели кружка: ставим 255 (белый фон).
    cleaned = crop.copy()
    cleaned[mask > 0] = 255

    circle_b64 = _np_png_b64(mask)

    return cleaned, {
        "circle_found": True,
        "circle_png_b64": circle_b64,
        "circle_bbox": [
            int(bbox[0]), int(bbox[1]),
            int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]),
        ],
        "circle_circularity": float(round(circularity, 3)),
        "note": "circle_removed",
    }


def _segment_digits(crop):
    """
    Разбивает кроп (65x65 или любой) на отдельные цифры
    по вертикальной проекции.

    Возвращает список (digit_mask, x_start, x_end),
    где digit_mask — uint8, цифра яркая на тёмном фоне,
    уже обрезанная по bbox с padding 2 px.
    """
    import numpy as np
    import cv2

    if crop.size == 0:
        return []

    inverted = 255 - crop

    _, binary = cv2.threshold(
        inverted, 0, 255,
        cv2.THRESH_BINARY | cv2.THRESH_OTSU,
    )

    profile = (binary > 0).sum(axis=0).astype(np.float32)

    if profile.max() <= 0:
        return []

    profile = np.convolve(profile, [0.25, 0.5, 0.25], mode="same")

    min_signal = max(1.0, profile.max() * 0.08)
    occupied = profile >= min_signal

    padded = np.pad(occupied.astype(np.int8), (1, 1))
    transitions = np.diff(padded)
    starts = np.where(transitions == 1)[0]
    ends = np.where(transitions == -1)[0]

    runs = [(int(a), int(b)) for a, b in zip(starts, ends)]

    if not runs:
        return []

    max_gap = max(2, int(round(binary.shape[1] * 0.04)))

    merged = []
    for a, b in runs:
        if not merged:
            merged.append([a, b])
            continue
        previous = merged[-1]
        gap = a - previous[1]
        if gap <= max_gap:
            previous[1] = b
        else:
            merged.append([a, b])

    min_width = max(2, int(round(binary.shape[1] * 0.03)))
    max_profile = float(profile.max())

    filtered = []
    for a, b in merged:
        width = b - a
        strength = float(profile[a:b].sum())
        if width < min_width:
            continue
        if strength < max_profile * 0.15:
            continue
        filtered.append((a, b))

    if not filtered:
        return []

    digits = []

    for a, b in filtered:
        margin = max(1, int(round((b - a) * 0.05)))
        left = max(0, a - margin)
        right = min(binary.shape[1], b + margin)

        fragment = inverted[:, left:right].copy()

        ys, xs = np.where(fragment > 32)
        if len(xs) == 0:
            continue

        y1, y2 = max(0, ys.min() - 2), min(fragment.shape[0], ys.max() + 3)
        x1, x2 = max(0, xs.min() - 2), min(fragment.shape[1], xs.max() + 3)

        if x2 - x1 < 2 or y2 - y1 < 2:
            continue

        tight = fragment[y1:y2, x1:x2].copy()

        digits.append((tight, left, right))

    return digits


def _mnist_predict_cell_with_debug(crop):
    """
    Убирает кружок (если есть), сегментирует кроп на отдельные цифры,
    прогоняет каждую через MNIST и конкатенирует результаты.

    Возвращает (digit_str, confidence_avg, debug).
    """
    import numpy as np

    if crop.size == 0:
        return None, None, None

    crop_b64 = _np_png_b64(crop)

    if float(crop.mean()) >= 250.0 and float(crop.std()) < 2.0:
        return None, None, {
            "crop_png_b64": crop_b64,
            "inverted_png_b64": "",
            "tight_png_b64": "",
            "tight_w": 0,
            "tight_h": 0,
            "digit_pngs_b64": [],
            "digit_texts": [],
            "circle_found": False,
            "circle_png_b64": "",
            "circle_bbox": None,
            "circle_circularity": 0.0,
            "circle_note": "",
            "note": "empty_crop",
        }

    # 1. Убираем кружок, если есть.
    cleaned, circle_debug = _remove_circle_mark(crop)

    inverted = 255 - cleaned
    inverted_b64 = _np_png_b64(inverted)

    # 2. Сегментируем очищенный кроп.
    digits = _segment_digits(cleaned)

    if not digits:
        return None, None, {
            "crop_png_b64": crop_b64,
            "inverted_png_b64": inverted_b64,
            "tight_png_b64": "",
            "tight_w": 0,
            "tight_h": 0,
            "digit_pngs_b64": [],
            "digit_texts": [],
            "circle_found": circle_debug["circle_found"],
            "circle_png_b64": circle_debug["circle_png_b64"],
            "circle_bbox": circle_debug["circle_bbox"],
            "circle_circularity": circle_debug["circle_circularity"],
            "circle_note": circle_debug["note"],
            "note": "no_digits",
        }

    recognized_text = []
    confidences = []
    digit_pngs = []
    digit_texts = []

    for tight, _left, _right in digits:
        try:
            char, conf, prepared = _mnist_digit(tight)
        except Exception:
            continue

        recognized_text.append(char)
        confidences.append(float(conf))
        digit_texts.append(char)
        digit_pngs.append(_np_png_b64(prepared))

    if not recognized_text:
        return None, None, {
            "crop_png_b64": crop_b64,
            "inverted_png_b64": inverted_b64,
            "tight_png_b64": "",
            "tight_w": 0,
            "tight_h": 0,
            "digit_pngs_b64": [],
            "digit_texts": [],
            "circle_found": circle_debug["circle_found"],
            "circle_png_b64": circle_debug["circle_png_b64"],
            "circle_bbox": circle_debug["circle_bbox"],
            "circle_circularity": circle_debug["circle_circularity"],
            "circle_note": circle_debug["note"],
            "note": "all_digits_failed",
        }

    # Превью «что ушло в MNIST»: полоса из 28×28 входов.
    strip = np.zeros((28, 28 * len(digits)), dtype=np.uint8)
    for i, (tight, _l, _r) in enumerate(digits):
        try:
            _, _, prepared = _mnist_digit(tight)
            strip[:, i * 28:(i + 1) * 28] = prepared
        except Exception:
            pass

    strip_b64 = _np_png_b64(strip)

    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

    return "".join(recognized_text), round(avg_conf, 3), {
        "crop_png_b64": crop_b64,
        "inverted_png_b64": inverted_b64,
        "tight_png_b64": strip_b64,
        "tight_w": int(strip.shape[1]),
        "tight_h": int(strip.shape[0]),
        "digit_pngs_b64": digit_pngs,
        "digit_texts": digit_texts,
        "circle_found": circle_debug["circle_found"],
        "circle_png_b64": circle_debug["circle_png_b64"],
        "circle_bbox": circle_debug["circle_bbox"],
        "circle_circularity": circle_debug["circle_circularity"],
        "circle_note": circle_debug["note"],
        "note": "ok",
    }


@app.post("/api/extract-cells")
async def extract_cells_endpoint(req: ExtractCellsRequest):
    if req.grid_size not in (9, 13, 19):
        raise HTTPException(422, "invalid grid size")

    if (
        len(req.x_lines) != req.grid_size
        or len(req.y_lines) != req.grid_size
    ):
        raise HTTPException(422, "invalid grid lines")

    if len(req.rectified_png_b64) > 60_000_000:
        raise HTTPException(413, "image payload too large")

    try:
        raw = base64.b64decode(
            req.rectified_png_b64,
            validate=True,
        )

        with Image.open(io.BytesIO(raw)) as source:
            if source.format != "PNG":
                raise ValueError("expected PNG image")

            if max(source.size) > 6000:
                raise ValueError("image is too large")

            image = source.convert("RGB")

    except (ValueError, binascii.Error, OSError) as e:
        raise HTTPException(400, f"invalid image: {e}")

    import numpy as np
    import cv2

    gray = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2GRAY)

    h, w = gray.shape

    dx = float(np.median(np.diff(req.x_lines)))
    dy = float(np.median(np.diff(req.y_lines)))

    if dx <= 0 or dy <= 0:
        raise HTTPException(422, "invalid grid spacing")

    half_x = max(8, int(round(dx * 0.45)))
    half_y = max(8, int(round(dy * 0.45)))

    def _build_cells():
        out = []

        for row in range(req.grid_size):
            for col in range(req.grid_size):
                dx_shift = 0
                dy_shift = 0

                cx = int(round(req.x_lines[col] + dx_shift * dx))
                cy = int(round(req.y_lines[row] - dy_shift * dy))

                x1 = max(0, cx - half_x)
                y1 = max(0, cy - half_y)
                x2 = min(w, cx + half_x + 1)
                y2 = min(h, cy + half_y + 1)

                if x2 <= x1 or y2 <= y1:
                    out.append({
                        "row": row,
                        "col": col,
                        "width": 0,
                        "height": 0,
                        "png_b64": "",
                        "mnist_digit": None,
                        "mnist_confidence": None,
                        "crop_png_b64": "",
                        "inverted_png_b64": "",
                        "tight_png_b64": "",
                        "tight_w": 0,
                        "tight_h": 0,
                        "digit_pngs_b64": [],
                        "digit_texts": [],
                        "circle_found": False,
                        "circle_png_b64": "",
                        "circle_bbox": None,
                        "circle_circularity": 0.0,
                        "circle_note": "",
                        "note": "empty",
                    })
                    continue

                crop = gray[y1:y2, x1:x2].copy()

                ok, png = cv2.imencode(".png", crop)
                if not ok:
                    raise HTTPException(500, "cannot encode cell")

                digit_str, conf, debug = _mnist_predict_cell_with_debug(crop)

                item = {
                    "row": row,
                    "col": col,
                    "width": int(crop.shape[1]),
                    "height": int(crop.shape[0]),
                    "png_b64": base64.b64encode(png.tobytes()).decode("ascii"),
                    "mnist_digit": digit_str,
                    "mnist_confidence": conf,
                }

                if debug is None:
                    item.update({
                        "crop_png_b64": "",
                        "inverted_png_b64": "",
                        "tight_png_b64": "",
                        "tight_w": 0,
                        "tight_h": 0,
                        "digit_pngs_b64": [],
                        "digit_texts": [],
                        "circle_found": False,
                        "circle_png_b64": "",
                        "circle_bbox": None,
                        "circle_circularity": 0.0,
                        "circle_note": "",
                        "note": "empty",
                    })
                else:
                    item.update(debug)

                out.append(item)

        return out

    cells = await run_in_threadpool(_build_cells)

    return {
        "grid_size": req.grid_size,
        "cells": cells,
    }


class GridMaskRequest(BaseModel):
    rectified_png_b64: str
    grid_size: int = 19
    x_lines: list[float]
    y_lines: list[float]
    line_width: int = 5
    mode: str = "lighten"
    lift_factor: float = 0.85


@app.post("/api/grid-mask")
async def grid_mask_endpoint(req: GridMaskRequest):
    if req.grid_size not in (9, 13, 19):
        raise HTTPException(422, "invalid grid size")

    if (
        len(req.x_lines) != req.grid_size
        or len(req.y_lines) != req.grid_size
    ):
        raise HTTPException(422, "invalid grid lines")

    if req.mode not in ("lighten", "white"):
        raise HTTPException(422, "invalid mode")

    if req.line_width < 1 or req.line_width > 50:
        raise HTTPException(422, "invalid line_width")

    if not 0.0 <= req.lift_factor <= 1.0:
        raise HTTPException(422, "invalid lift_factor")

    if len(req.rectified_png_b64) > 60_000_000:
        raise HTTPException(413, "image payload too large")

    try:
        raw = base64.b64decode(
            req.rectified_png_b64,
            validate=True,
        )

        with Image.open(io.BytesIO(raw)) as source:
            if source.format != "PNG":
                raise ValueError("expected PNG image")

            if max(source.size) > 6000:
                raise ValueError("image is too large")

            image = source.convert("RGB")

    except (ValueError, binascii.Error, OSError) as e:
        raise HTTPException(400, f"invalid image: {e}")

    import numpy as np
    import cv2

    rgb = np.asarray(image).copy()
    h, w = rgb.shape[:2]

    mask = np.zeros((h, w), dtype=np.uint8)

    for x in req.x_lines:
        xi = int(round(x))
        if 0 <= xi < w:
            cv2.line(mask, (xi, 0), (xi, h - 1), 255, req.line_width)

    for y in req.y_lines:
        yi = int(round(y))
        if 0 <= yi < h:
            cv2.line(mask, (0, yi), (w - 1, yi), 255, req.line_width)

    if req.mode == "white":
        rgb[mask > 0] = 255
    else:
        lifted = (
            rgb.astype(np.float32)
            + (255.0 - rgb.astype(np.float32)) * req.lift_factor
        ).astype(np.uint8)
        rgb[mask > 0] = lifted[mask > 0]

    out = Image.fromarray(rgb, mode="RGB")

    return {
        "width": w,
        "height": h,
        "masked_png_b64": _png_b64(out),
        "mode": req.mode,
        "line_width": req.line_width,
        "lift_factor": req.lift_factor,
    }