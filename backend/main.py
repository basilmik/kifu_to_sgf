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


def _detect_circle(crop):
    """
    Ищет контур кружка/овала, обводящего ход.

    Возвращает debug-dict:
        "circle_found": bool,
        "circle_mask_png_b64": маска кружка (белый контур на чёрном),
        "circle_overlay_png_b64": исходный кроп с красным контуром (BGR),
        "circle_bbox": [x1, y1, x2, y2] или None,
        "circle_circularity": float,
        "circle_note": str
    """
    import numpy as np
    import cv2

    empty = {
        "circle_found": False,
        "circle_mask_png_b64": "",
        "circle_overlay_png_b64": "",
        "circle_bbox": None,
        "circle_circularity": 0.0,
        "circle_note": "",
    }

    if crop.size == 0:
        return empty

    h, w = crop.shape

    inverted = 255 - crop
    _, binary = cv2.threshold(
        inverted, 0, 255,
        cv2.THRESH_BINARY | cv2.THRESH_OTSU,
    )

    contours, _ = cv2.findContours(
        binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        empty["circle_note"] = "no_contours"
        return empty

    crop_area = float(h * w)
    min_bbox_side = 0.35 * min(h, w)

    best = None
    best_score = 0.0

    for c in contours:
        area = float(cv2.contourArea(c))
        if area <= 0:
            continue

        peri = float(cv2.arcLength(c, True))
        if peri <= 0:
            continue

        x, y, bw, bh = cv2.boundingRect(c)

        if bw < min_bbox_side or bh < min_bbox_side:
            continue

        aspect = max(bw, bh) / max(1.0, min(bw, bh))
        if aspect > 1.8:
            continue

        circularity = 4.0 * math.pi * area / (peri * peri)

        bbox_area = float(bw * bh)
        extent = area / bbox_area if bbox_area > 0 else 0.0

        if circularity < 0.45:
            continue

        if bbox_area < 0.30 * crop_area:
            continue

        if extent > 0.7:
            continue

        score = circularity + (1.0 - extent) * 0.5
        if score > best_score:
            best_score = score
            best = (c, area, peri, circularity, extent, (x, y, bw, bh))

    if best is None:
        empty["circle_note"] = "no_circle_candidate"
        return empty

    c, area, peri, circularity, extent, bbox = best

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(mask, [c], -1, 255, thickness=1)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)

    mask_b64 = _np_png_b64(mask)

    overlay = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(overlay, [c], -1, (0, 0, 255), 1)
    overlay_b64 = _np_png_b64(overlay)

    return {
        "circle_found": True,
        "circle_mask_png_b64": mask_b64,
        "circle_overlay_png_b64": overlay_b64,
        "circle_bbox": [
            int(bbox[0]), int(bbox[1]),
            int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]),
        ],
        "circle_circularity": float(round(circularity, 3)),
        "circle_note": "circle_detected",
    }


# def _segment_digits(crop):
#     """
#     Разбивает кроп (65x65 или любой) на отдельные цифры
#     по вертикальной проекции.

#     Возвращает список (digit_mask, x_start, x_end).
#     """
#     import numpy as np
#     import cv2

#     if crop.size == 0:
#         return []

#     inverted = 255 - crop

#     _, binary = cv2.threshold(
#         inverted, 0, 255,
#         cv2.THRESH_BINARY | cv2.THRESH_OTSU,
#     )

#     profile = (binary > 0).sum(axis=0).astype(np.float32)

#     if profile.max() <= 0:
#         return []

#     profile = np.convolve(profile, [0.25, 0.5, 0.25], mode="same")

#     min_signal = max(1.0, profile.max() * 0.08)
#     occupied = profile >= min_signal

#     padded = np.pad(occupied.astype(np.int8), (1, 1))
#     transitions = np.diff(padded)
#     starts = np.where(transitions == 1)[0]
#     ends = np.where(transitions == -1)[0]

#     runs = [(int(a), int(b)) for a, b in zip(starts, ends)]

#     if not runs:
#         return []

#     max_gap = max(2, int(round(binary.shape[1] * 0.04)))

#     merged = []
#     for a, b in runs:
#         if not merged:
#             merged.append([a, b])
#             continue
#         previous = merged[-1]
#         gap = a - previous[1]
#         if gap <= max_gap:
#             previous[1] = b
#         else:
#             merged.append([a, b])

#     min_width = max(2, int(round(binary.shape[1] * 0.03)))
#     max_profile = float(profile.max())

#     filtered = []
#     for a, b in merged:
#         width = b - a
#         strength = float(profile[a:b].sum())
#         if width < min_width:
#             continue
#         if strength < max_profile * 0.15:
#             continue
#         filtered.append((a, b))

#     if not filtered:
#         return []

#     digits = []

#     for a, b in filtered:
#         margin = max(1, int(round((b - a) * 0.05)))
#         left = max(0, a - margin)
#         right = min(binary.shape[1], b + margin)

#         fragment = inverted[:, left:right].copy()

#         ys, xs = np.where(fragment > 32)
#         if len(xs) == 0:
#             continue

#         y1, y2 = max(0, ys.min() - 2), min(fragment.shape[0], ys.max() + 3)
#         x1, x2 = max(0, xs.min() - 2), min(fragment.shape[1], xs.max() + 3)

#         if x2 - x1 < 2 or y2 - y1 < 2:
#             continue

#         tight = fragment[y1:y2, x1:x2].copy()

#         digits.append((tight, left, right))

#     return digits

def _has_handwriting(crop,  is_hoshi=False):
    """
    Проверка наличия заметных штрихов после подавления сетки.
    Используется только для отсеивания пустых ROI.
    """
    import cv2
    import numpy as np

    if crop.size == 0:
        return False

    gray = crop.astype(np.float32)
    paper = float(np.percentile(gray, 60))

    ink = np.clip(
        paper - gray,
        0,
        255,
    )

    h, w = ink.shape
    edge_y = max(1, round(h * 0.18))
    edge_x = max(1, round(w * 0.18))

    # Оцениваем стороны отдельно.
    top = np.median(ink[:edge_y], axis=0)
    bottom = np.median(ink[-edge_y:], axis=0)

    # Если линия есть только с одной стороны,
    # берём её полный контраст.
    vertical = np.maximum(
        top,
        bottom,
    ).astype(np.float32)

    # Допуск в один пиксель на смещение линии.
    vertical = cv2.dilate(
        vertical[None, :],
        np.ones((1, 3), dtype=np.uint8),
    )[0]

    probe = np.clip(
        ink - vertical[None, :],
        0,
        255,
    )

    left = np.median(probe[:, :edge_x], axis=1)
    right = np.median(probe[:, -edge_x:], axis=1)

    horizontal = np.maximum(
        left,
        right,
    ).astype(np.float32)

    horizontal = cv2.dilate(
        horizontal[:, None],
        np.ones((3, 1), dtype=np.uint8),
    )[:, 0]

    probe = np.clip(
        probe - horizontal[:, None],
        0,
        255,
    ).astype(np.uint8)

    otsu, _ = cv2.threshold(
        probe,
        0,
        255,
        cv2.THRESH_BINARY | cv2.THRESH_OTSU,
    )

    binary = (
        probe > max(10.0, otsu)
    ).astype(np.uint8)

    count, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            binary,
            connectivity=8,
        )
    )

    min_area = max(
        3,
        round(crop.size * 0.0006),
    )

    # Собираем все значимые компоненты.
    valid_ids = []

    for component_id in range(1, count):
        if (
            stats[component_id, cv2.CC_STAT_AREA]
            < min_area
        ):
            continue

        if (
            stats[component_id, cv2.CC_STAT_HEIGHT]
            < 3
        ):
            continue

        peak = int(
            probe[labels == component_id].max()
        )

        if peak >= 30:
            valid_ids.append(component_id)

    if not valid_ids:
        return False

    # Проверяем именно пустое пересечение с точкой хоси.
    if is_hoshi:
        selected = np.isin(labels, valid_ids)
        yy, xx = np.where(selected)

        x1 = int(xx.min())
        x2 = int(xx.max())
        y1 = int(yy.min())
        y2 = int(yy.max())

        mark_w = x2 - x1 + 1
        mark_h = y2 - y1 + 1

        roi_size = min(h, w)

        # Для ROI 65×65 — максимум около 10×10 пикселей.
        max_dot_size = max(
            4,
            round(roi_size * 0.16),
        )

        # Допускаем небольшое смещение относительно центра.
        center_tolerance = max(
            2,
            round(roi_size * 0.06),
        )

        mark_cx = (x1 + x2) / 2.0
        mark_cy = (y1 + y2) / 2.0

        roi_cx = (w - 1) / 2.0
        roi_cy = (h - 1) / 2.0

        small = (
            mark_w <= max_dot_size
            and mark_h <= max_dot_size
        )

        compact = (
            0.5 <= mark_w / max(1, mark_h) <= 2.0
        )

        centered = (
            abs(mark_cx - roi_cx) <= center_tolerance
            and abs(mark_cy - roi_cy) <= center_tolerance
        )

        if small and compact and centered:
            return False

    return True


def _segment_digits(crop,  is_hoshi=False):
    """
    Вход: grayscale ROI, тёмные цифры на светлом фоне.

    Выход:
        digits: [(digit_mask, x_start, x_end), ...]
        status: строка для диагностики

    digit_mask: светлые штрихи на чёрном фоне.
    """
    import cv2
    import numpy as np

    if crop.size == 0:
        return [], "empty_crop"

    if not _has_handwriting(
        crop,
        is_hoshi=is_hoshi,
    ):
        return [], "blank_or_grid"
    # 1. Контраст относительно бумаги.
    gray = crop.astype(np.float32)
    paper = float(np.percentile(gray, 60))

    ink = np.clip(
        paper - gray,
        0,
        255,
    ).astype(np.uint8)

    h, w = ink.shape

    # 2. Оценка вертикального фона по верхнему
    # и нижнему краям ROI.
    edge_y = max(1, round(h * 0.18))
    edge_x = max(1, round(w * 0.18))

    top_bottom = np.concatenate(
        (ink[:edge_y], ink[-edge_y:]),
        axis=0,
    )

    vertical_background = np.median(
        top_bottom,
        axis=0,
    )

    ink = np.clip(
        ink.astype(np.float32) - vertical_background,
        0,
        255,
    )

    # 3. Оценка горизонтального фона по левому
    # и правому краям ROI.
    left_right = np.concatenate(
        (ink[:, :edge_x], ink[:, -edge_x:]),
        axis=1,
    )

    horizontal_background = np.median(
        left_right,
        axis=1,
    )

    ink = np.clip(
        ink - horizontal_background[:, None],
        0,
        255,
    ).astype(np.uint8)

    # 4. Маска для поиска компонентов.
    otsu, _ = cv2.threshold(
        ink,
        0,
        255,
        cv2.THRESH_BINARY | cv2.THRESH_OTSU,
    )

    binary = (
        ink > max(10.0, otsu)
    ).astype(np.uint8)

    count, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            binary,
            connectivity=8,
        )
    )

    # Фильтруем по площади и наличию достаточно
    # контрастного участка.
    # Поэтому узкая единица может сохраниться.
    min_area = max(
        2,
        round(crop.size * 0.0006),
    )

    component_ids = [
        i
        for i in range(1, count)
        if stats[i, cv2.CC_STAT_AREA] >= min_area
        and int(ink[labels == i].max()) >= 30
    ]

    if not component_ids:
        return [], "blank_or_faint"

    # 5. Объединение вертикально разорванных штрихов.
    parent = {i: i for i in component_ids}

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    max_vertical_gap = max(
        2,
        round(h * 0.12),
    )

    def stroke_ends(component_id):
        yy, xx = np.where(labels == component_id)

        _, y, _, height, _ = map(
            int,
            stats[component_id],
        )

        # Берём небольшие участки сверху и снизу компоненты.
        band = max(1, round(height * 0.20))

        top = yy < y + band
        bottom = yy >= y + height - band

        return (
            (
                float(np.median(xx[top])),
                float(np.median(yy[top])),
            ),
            (
                float(np.median(xx[bottom])),
                float(np.median(yy[bottom])),
            ),
        )


    ends = {
        component_id: stroke_ends(component_id)
        for component_id in component_ids
    }


    for index, a in enumerate(component_ids):
        ax, ay, aw, ah, _ = map(int, stats[a])

        for b in component_ids[index + 1:]:
            bx, by, bw, bh, _ = map(int, stats[b])

            overlap_x = max(
                0,
                min(ax + aw, bx + bw) - max(ax, bx),
            )

            overlap_y = max(
                0,
                min(ay + ah, by + bh) - max(ay, by),
            )

            gap_y = max(
                0,
                max(ay, by) - min(ay + ah, by + bh),
            )

            # Прежнее правило: части расположены одна над другой
            # и достаточно перекрываются по X.
            aligned_fragments = (
                overlap_x >= 0.45 * min(aw, bw)
                and overlap_y <= 0.20 * min(ah, bh)
                and gap_y <= max_vertical_gap
            )

            # Дополнительное правило для разорванных
            # вертикальных и наклонённых штрихов.
            continued_stroke = False

            if (
                overlap_y == 0
                and gap_y <= max_vertical_gap
                and aw <= 0.85 * ah
                and bw <= 0.85 * bh
            ):
                upper, lower = (
                    (a, b) if ay < by else (b, a)
                )

                upper_top, upper_bottom = ends[upper]
                lower_top, lower_bottom = ends[lower]

                upper_slope = (
                    (upper_bottom[0] - upper_top[0])
                    / max(1.0, upper_bottom[1] - upper_top[1])
                )

                lower_slope = (
                    (lower_bottom[0] - lower_top[0])
                    / max(1.0, lower_bottom[1] - lower_top[1])
                )

                # Продолжаем верхний штрих до высоты
                # начала нижнего.
                predicted_x = (
                    upper_bottom[0]
                    + upper_slope
                    * (lower_top[1] - upper_bottom[1])
                )

                continued_stroke = (
                    abs(upper_slope - lower_slope) <= 0.35
                    and abs(predicted_x - lower_top[0])
                    <= max(2.0, w * 0.04)
                )

            if aligned_fragments or continued_stroke:
                parent[root(b)] = root(a)

    groups = {}

    for component_id in component_ids:
        groups.setdefault(
            root(component_id),
            [],
        ).append(component_id)

    if len(groups) > 1:
        boxes = {}

        for group_id, group_ids in groups.items():
            yy, xx = np.where(
                np.isin(labels, group_ids)
            )

            boxes[group_id] = (
                int(xx.min()),
                int(yy.min()),
                int(xx.max()) + 1,
                int(yy.max()) + 1,
                len(xx),
            )

        reference_h = max(
            y2 - y1
            for x1, y1, x2, y2, area in boxes.values()
        )

        reference_area = max(
            box[4] for box in boxes.values()
        )

        tiny_groups = {
            group_id
            for group_id, (
                x1, y1, x2, y2, area
            ) in boxes.items()
            if (
                y2 - y1 <= 0.25 * reference_h
                and area <= 0.15 * reference_area
            )
        }

        for group_id in tiny_groups:
            x1, y1, x2, y2, _ = boxes[group_id]

            targets = []

            for target_id, (
                tx1, ty1, tx2, ty2, _
            ) in boxes.items():

                if target_id in tiny_groups:
                    continue

                # Маленький фрагмент целиком находится
                # внутри прямоугольника большой группы.
                if (
                    tx1 <= x1
                    and ty1 <= y1
                    and x2 <= tx2
                    and y2 <= ty2
                ):
                    targets.append(target_id)

            # Объединяем только при единственном
            # подходящем владельце.
            if len(targets) == 1:
                groups[targets[0]].extend(
                    groups.pop(group_id)
                )

    # Присоединяем короткие соседние фрагменты:
    # кончики, хвостики и другие отделившиеся части цифры.
    if len(groups) > 1:
        boxes = {}
        masks = {}

        for group_id, group_ids in groups.items():
            mask = np.isin(labels, group_ids)
            yy, xx = np.where(mask)

            masks[group_id] = mask

            boxes[group_id] = (
                int(xx.min()),
                int(yy.min()),
                int(xx.max()) + 1,
                int(yy.max()) + 1,
                len(xx),
            )

        reference_h = max(
            y2 - y1
            for x1, y1, x2, y2, area in boxes.values()
        )

        reference_area = max(
            box[4] for box in boxes.values()
        )

        # Короткие и небольшие группы считаем
        # возможными отделившимися фрагментами.
        small_groups = {
            group_id
            for group_id, (
                x1, y1, x2, y2, area
            ) in boxes.items()
            if (
                y2 - y1 <= 0.35 * reference_h
                and area <= 0.40 * reference_area
            )
        }

        large_groups = set(groups) - small_groups

        # Для каждой основной группы строим карту
        # расстояний до её реальных пикселей.
        distance_maps = {
            group_id: cv2.distanceTransform(
                (~masks[group_id]).astype(np.uint8),
                cv2.DIST_L2,
                cv2.DIST_MASK_PRECISE,
            )
            for group_id in large_groups
        }

        for group_id in small_groups:
            x1, y1, x2, y2, area = boxes[group_id]
            small_h = y2 - y1

            candidates = []

            for target_id in large_groups:
                (
                    tx1, ty1, tx2, ty2, target_area
                ) = boxes[target_id]

                target_h = ty2 - ty1

                # Фрагмент должен быть небольшим
                # относительно именно этого соседа.
                if (
                    small_h > 0.35 * target_h
                    or area > 0.40 * target_area
                ):
                    continue

                gap = float(
                    distance_maps[target_id][
                        masks[group_id]
                    ].min()
                )

                max_gap = max(
                    2.0,
                    0.10 * target_h,
                )

                if gap <= max_gap:
                    candidates.append(target_id)

            # При нескольких подходящих соседях
            # сохраняем фрагмент отдельно.
            if len(candidates) == 1:
                target_id = candidates[0]

                groups[target_id].extend(
                    groups.pop(group_id)
                )

    # 6. Изолируем пиксели каждой группы.
    digits = []

    for group_ids in groups.values():
        selected = np.isin(labels, group_ids)

        yy, xx = np.where(selected)

        x1 = int(xx.min())
        x2 = int(xx.max()) + 1
        y1 = int(yy.min())
        y2 = int(yy.max()) + 1

        fragment = np.where(
            selected,
            ink,
            0,
        ).astype(np.uint8)

        tight = fragment[y1:y2, x1:x2].copy()

        digits.append((tight, x1, x2))

    # Для наклонённых цифр диапазоны X могут
    # перекрываться. Сортируем по центрам.
    digits.sort(
        key=lambda item: (item[1] + item[2]) / 2,
    )

    if len(digits) > 3:
        return [], f"uncertain_split:{len(digits)}"

    return digits, "ok"


def _mnist_predict_cell_with_debug(crop, is_hoshi=False):
    """
    Ищет кружок (только для отображения, НЕ удаляет его),
    сегментирует исходный кроп на отдельные цифры,
    прогоняет каждую через MNIST и конкатенирует результаты.

    Возвращает (digit_str, confidence_avg, debug).
    """
    import numpy as np

    if crop.size == 0:
        return None, None, None

    crop_b64 = _np_png_b64(crop)

    empty_circle = {
        "circle_found": False,
        "circle_mask_png_b64": "",
        "circle_overlay_png_b64": "",
        "circle_bbox": None,
        "circle_circularity": 0.0,
        "circle_note": "",
    }

    if float(crop.mean()) >= 250.0 and float(crop.std()) < 2.0:
        return None, None, {
            "crop_png_b64": crop_b64,
            "inverted_png_b64": "",
            "tight_png_b64": "",
            "tight_w": 0,
            "tight_h": 0,
            "digit_pngs_b64": [],
            "digit_texts": [],
            **empty_circle,
            "note": "empty_crop",
        }

    # 1. Ищем кружок. НЕ удаляем — только для отображения.
    circle_debug = _detect_circle(crop)

    # 2. Сегментируем исходный кроп (как раньше).
    inverted = 255 - crop
    inverted_b64 = _np_png_b64(inverted)

    digits, segmentation_note = _segment_digits(crop, is_hoshi)

    if not digits:
        return None, None, {
            "crop_png_b64": crop_b64,
            "inverted_png_b64": inverted_b64,
            "tight_png_b64": "",
            "tight_w": 0,
            "tight_h": 0,
            "digit_pngs_b64": [],
            "digit_texts": [],
            **circle_debug,
            "note": segmentation_note,
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
            **circle_debug,
            "note": "all_digits_failed",
        }

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
        **circle_debug,
        "note": segmentation_note,
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
                cx = int(round(req.x_lines[col]))
                cy = int(round(req.y_lines[row]))

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
                        "circle_mask_png_b64": "",
                        "circle_overlay_png_b64": "",
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

                is_hoshi = (
                    req.grid_size == 19
                    and row in (3, 9, 15)
                    and col in (3, 9, 15)
                )

                digit_str, conf, debug = (
                    _mnist_predict_cell_with_debug(
                        crop,
                        is_hoshi=is_hoshi,
                    )
                )

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
                        "circle_mask_png_b64": "",
                        "circle_overlay_png_b64": "",
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