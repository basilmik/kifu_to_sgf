from __future__ import annotations

import base64
from functools import lru_cache
from pathlib import Path
from threading import Lock
from time import perf_counter

import cv2
import numpy as np
from PIL import Image


MODELS = Path(__file__).resolve().parent / "models"
_LOCK = Lock()


def _png_b64(image: np.ndarray) -> str:
    ok, encoded = cv2.imencode(".png", image)

    if not ok:
        raise RuntimeError("PNG encode failed")

    return base64.b64encode(
        encoded.tobytes()
    ).decode("ascii")


@lru_cache(maxsize=1)
def _mnist_session():
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1

    return ort.InferenceSession(
        str(MODELS / "mnist-8.onnx"),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )


@lru_cache(maxsize=1)
def _crnn_net():
    net = cv2.dnn.readNetFromONNX(
        str(
            MODELS /
            "text_recognition_CRNN_EN_2021sep.onnx"
        )
    )

    net.setPreferableBackend(
        cv2.dnn.DNN_BACKEND_OPENCV
    )

    net.setPreferableTarget(
        cv2.dnn.DNN_TARGET_CPU
    )

    return net


def _mask_and_digits(
    gray: np.ndarray,
    cx: int,
    cy: int,
    pitch: float,
):
    blurred = cv2.GaussianBlur(
        gray, (3, 3), 0
    )

    _, mask = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU,
    )

    # Удаляем предполагаемые линии сетки.
    # Только в ветке сегментации: исходный ROI
    # для CRNN не изменяем.
    line_width = max(
        1, int(round(pitch * 0.025))
    )

    cv2.line(
        mask,
        (cx, 0),
        (cx, mask.shape[0] - 1),
        0,
        line_width,
    )

    cv2.line(
        mask,
        (0, cy),
        (mask.shape[1] - 1, cy),
        0,
        line_width,
    )

    # Частично соединяем разорванные штрихи.
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        np.ones((2, 2), np.uint8),
    )

    # Убираем мелкий шум.
    count, labels, stats, _ = (
        cv2.connectedComponentsWithStats(mask, 8)
    )

    cleaned = np.zeros_like(mask)

    min_area = max(
        4, int(pitch * pitch * 0.002)
    )

    for i in range(1, count):
        x, y, w, h, area = stats[i]

        if (
            area >= min_area
            and h >= 4
            and w <= mask.shape[1] * 0.95
        ):
            cleaned[labels == i] = 255

    ys, xs = np.where(cleaned > 0)

    if not len(xs):
        return cleaned, [], "blank"

    min_x = int(xs.min())
    max_x = int(xs.max()) + 1

    min_y = int(ys.min())
    max_y = int(ys.max()) + 1

    tight = cleaned[
        min_y:max_y,
        min_x:max_x,
    ]

    # Вертикальная проекция.
    occupied = np.any(
        tight > 0, axis=0
    ).astype(np.int8)

    padded = np.pad(occupied, (1, 1))

    starts = np.where(
        np.diff(padded) == 1
    )[0]

    ends = np.where(
        np.diff(padded) == -1
    )[0]

    runs = list(zip(starts, ends))

    # Для этого эксперимента допускаем
    # от 1 до 3 отдельных цифр.
    if not 1 <= len(runs) <= 3:
        return (
            cleaned,
            [],
            f"uncertain_split:{len(runs)}",
        )

    digits = []

    for a, b in runs:
        fragment = tight[:, a:b]

        yy, xx = np.where(fragment > 0)

        if not len(xx):
            return cleaned, [], "empty_segment"

        digit = fragment[
            yy.min():yy.max() + 1,
            xx.min():xx.max() + 1,
        ]

        digits.append(digit)

    return cleaned, digits, "ok"


def _mask_and_digits(
    gray: np.ndarray,
    cx: int,
    cy: int,
    pitch: float,
):
    blurred = cv2.GaussianBlur(
        gray, (3, 3), 0
    )

    _, mask = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU,
    )

    # Удаляем предполагаемые линии сетки.
    # Только в ветке сегментации: исходный ROI
    # для CRNN не изменяем.
    line_width = max(
        1, int(round(pitch * 0.025))
    )

    cv2.line(
        mask,
        (cx, 0),
        (cx, mask.shape[0] - 1),
        0,
        line_width,
    )

    cv2.line(
        mask,
        (0, cy),
        (mask.shape[1] - 1, cy),
        0,
        line_width,
    )

    # Частично соединяем разорванные штрихи.
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        np.ones((2, 2), np.uint8),
    )

    # Убираем мелкий шум.
    count, labels, stats, _ = (
        cv2.connectedComponentsWithStats(mask, 8)
    )

    cleaned = np.zeros_like(mask)

    min_area = max(
        4, int(pitch * pitch * 0.002)
    )

    for i in range(1, count):
        x, y, w, h, area = stats[i]

        if (
            area >= min_area
            and h >= 4
            and w <= mask.shape[1] * 0.95
        ):
            cleaned[labels == i] = 255

    ys, xs = np.where(cleaned > 0)

    if not len(xs):
        return cleaned, [], "blank"

    min_x = int(xs.min())
    max_x = int(xs.max()) + 1

    min_y = int(ys.min())
    max_y = int(ys.max()) + 1

    tight = cleaned[
        min_y:max_y,
        min_x:max_x,
    ]

    # Вертикальная проекция.
    occupied = np.any(
        tight > 0, axis=0
    ).astype(np.int8)

    padded = np.pad(occupied, (1, 1))

    starts = np.where(
        np.diff(padded) == 1
    )[0]

    ends = np.where(
        np.diff(padded) == -1
    )[0]

    runs = list(zip(starts, ends))

    # Для этого эксперимента допускаем
    # от 1 до 3 отдельных цифр.
    if not 1 <= len(runs) <= 3:
        return (
            cleaned,
            [],
            f"uncertain_split:{len(runs)}",
        )

    digits = []

    for a, b in runs:
        fragment = tight[:, a:b]

        yy, xx = np.where(fragment > 0)

        if not len(xx):
            return cleaned, [], "empty_segment"

        digit = fragment[
            yy.min():yy.max() + 1,
            xx.min():xx.max() + 1,
        ]

        digits.append(digit)

    return cleaned, digits, "ok"


def _crnn_text(gray: np.ndarray) -> str:
    # Модель OpenCV Zoo ожидает 100x32.
    blob = cv2.dnn.blobFromImage(
        gray,
        scalefactor=1 / 127.5,
        size=(100, 32),
        mean=(127.5,),
        swapRB=False,
        crop=False,
    )

    net = _crnn_net()

    net.setInput(blob)
    logits = net.forward()

    # Формат выхода этой модели:
    # [time_steps, 1, 37]
    # 0 — CTC blank,
    # 1..36 — цифры и буквы.
    if (
        logits.ndim != 3
        or logits.shape[1] != 1
        or logits.shape[2] != 37
    ):
        raise RuntimeError(
            f"Unexpected CRNN output: {logits.shape}"
        )

    alphabet = (
        "0123456789abcdefghijklmnopqrstuvwxyz"
    )

    previous = -1
    characters = []

    for class_id in logits[:, 0, :].argmax(axis=1):
        cls = int(class_id)

        if cls != 0 and cls != previous:
            characters.append(
                alphabet[cls - 1]
            )

        previous = cls

    return "".join(characters)



def compare_digit_models(
    rectified: Image.Image,
    x_lines: list[float],
    y_lines: list[float],
    max_rois: int = 6,
    cells: list[tuple[int, int]] | None = None,
) -> dict:
    n = len(x_lines)

    if n not in (9, 13, 19) or len(y_lines) != n:
        raise ValueError("invalid grid")

    for lines in (x_lines, y_lines):
        if not all(np.isfinite(lines)):
            raise ValueError("non-finite grid coordinates")

        if any(
            b <= a for a, b in zip(lines, lines[1:])
        ):
            raise ValueError("invalid grid spacing")

    if not 1 <= max_rois <= 12:
        raise ValueError("max_rois must be in 1..12")

    if cells is not None:
        if not cells or len(cells) > max_rois:
            raise ValueError("invalid cells")

    gray = cv2.cvtColor(
        np.asarray(rectified.convert("RGB")),
        cv2.COLOR_RGB2GRAY,
    )

    h, w = gray.shape

    if (
        x_lines[0] < 0
        or x_lines[-1] >= w
        or y_lines[0] < 0
        or y_lines[-1] >= h
    ):
        raise ValueError("grid outside image")

    pitch = min(
        float(np.median(np.diff(x_lines))),
        float(np.median(np.diff(y_lines))),
    )

    radius = max(8, round(0.47 * pitch))

    # По умолчанию рассматриваем всю сетку.
    # Но можно передать конкретные пересечения.
    places = (
        cells
        if cells is not None
        else [
            (r, c)
            for r in range(n)
            for c in range(n)
        ]
    )

    proposals = []

    for row, col in places:
        if not 0 <= row < n or not 0 <= col < n:
            raise ValueError("cell outside grid")

        cx = round(x_lines[col])
        cy = round(y_lines[row])

        x1 = max(0, cx - radius)
        y1 = max(0, cy - radius)
        x2 = min(w, cx + radius + 1)
        y2 = min(h, cy + radius + 1)

        # Один исходный ROI для обоих методов.
        roi = gray[y1:y2, x1:x2].copy()

        cleaned, digits, status = _mask_and_digits(
            roi,
            cx - x1,
            cy - y1,
            pitch,
        )

        ink_score = (
            cv2.countNonZero(cleaned) / cleaned.size
        )

        proposals.append((
            ink_score,
            row,
            col,
            roi,
            cleaned,
            digits,
            status,
        ))

    # Если конкретные пересечения не выбраны,
    # берём области с наибольшим остатком чернил.
    if cells is None:
        proposals.sort(
            reverse=True,
            key=lambda item: item[0],
        )
        proposals = proposals[:max_rois]

    comparisons = []

    # Защита от одновременного запуска
    # тяжёлых сравнений в одном процессе.
    with _LOCK:
        started = perf_counter()

        _mnist_session()
        _crnn_net()

        for (
            ink_score,
            row,
            col,
            roi,
            cleaned,
            digits,
            status,
        ) in proposals:

            # Подход A: сегментация + MNIST.
            recognized = []
            scores = []
            digit_images = []

            if status == "ok":
                for digit in digits:
                    char, conf, mnist_img = (
                        _mnist_digit(digit)
                    )

                    recognized.append(char)
                    scores.append(conf)

                    digit_images.append(
                        _png_b64(255 - mnist_img)
                    )

            # Подход B: CRNN на исходном ROI.
            crnn_text = _crnn_text(roi)

            comparisons.append({
                "row": row,
                "col": col,
                "ink_score": round(ink_score, 4),

                "roi_png_b64": _png_b64(roi),

                "segmentation_png_b64": (
                    _png_b64(255 - cleaned)
                ),

                "digit_png_b64": digit_images,

                "mnist": {
                    "text": (
                        "".join(recognized)
                        if recognized else None
                    ),
                    "confidence_per_digit": scores,
                    "status": status,
                },

                "crnn": {
                    "text": crnn_text,
                    "valid_number": (
                        crnn_text.isdecimal()
                        and 1 <= len(crnn_text) <= 3
                    ),
                },
            })

        elapsed_ms = round(
            (perf_counter() - started) * 1000
        )

    return {
        "comparisons": comparisons,
        "elapsed_ms": elapsed_ms,
    }