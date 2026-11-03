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


def _center_of_mass(img: np.ndarray) -> tuple[float, float]:
    ys, xs = np.indices(img.shape)
    total = float(img.sum())

    if total <= 0.0:
        return img.shape[0] / 2.0, img.shape[1] / 2.0

    cy = float((ys * img).sum()) / total
    cx = float((xs * img).sum()) / total
    return cy, cx


def _prepare_mnist_input(mask: np.ndarray) -> np.ndarray:
    """
    Приводит маску цифры (uint8, цифра яркая на тёмном фоне)
    к виду, на котором обучалась onnxmodelzoo/mnist-8:

      - 28x28, grayscale;
      - цифра белая на чёрном фоне;
      - бинарные штрихи (0 / 255), без серости;
      - цифра вписана в 20x20 с сохранением пропорций;
      - центрирована по центру масс в 28x28.

    Возвращает uint8 (28, 28).
    """
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)

    if mask.size == 0:
        raise ValueError("Empty digit image")

    # 1. Бинаризация: оставляем только чёткие штрихи.
    _, binary = cv2.threshold(
        mask, 0, 255,
        cv2.THRESH_BINARY | cv2.THRESH_OTSU,
    )

    # 2. Обрезка по bbox с запасом 2 px,
    #    чтобы не отрезать края у 8/0/6/9.
    ys, xs = np.where(binary > 0)
    if len(xs) == 0:
        raise ValueError("Empty after threshold")

    h, w = binary.shape
    y1 = max(0, int(ys.min()) - 2)
    y2 = min(h, int(ys.max()) + 3)
    x1 = max(0, int(xs.min()) - 2)
    x2 = min(w, int(xs.max()) + 3)

    binary = binary[y1:y2, x1:x2]

    h, w = binary.shape
    if h == 0 or w == 0:
        raise ValueError("Empty after crop")

    # 3. Вписать в 20x20, сохраняя пропорции.
    scale = 20.0 / max(h, w)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    scaled = cv2.resize(
        binary,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA,
    )

    # 4. Вставить в центр 28x28.
    img = np.zeros((28, 28), dtype=np.uint8)
    oy = (28 - new_h) // 2
    ox = (28 - new_w) // 2
    img[oy:oy + new_h, ox:ox + new_w] = scaled

    # 5. Центрирование по центру масс.
    cy, cx = _center_of_mass(img)
    shift_y = 14.0 - cy
    shift_x = 14.0 - cx
    M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    img = cv2.warpAffine(
        img, M, (28, 28),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    return img


def _mnist_digit(mask: np.ndarray):
    """
    mask: uint8, цифра яркая на тёмном фоне.

    Возвращает (digit_str, confidence, prepared_28x28_uint8).
    """
    prepared = _prepare_mnist_input(mask)

    # onnxmodelzoo/mnist-8 ожидает float32 [1,1,28,28], x/255.
    # Никаких mean/std — они не нужны для этой модели.
    tensor = (
        prepared.astype(np.float32) / 255.0
    )[None, None, ...]

    session = _mnist_session()
    input_name = session.get_inputs()[0].name

    scores = session.run(
        None,
        {input_name: tensor},
    )[0].reshape(-1)

    if scores.size != 10:
        raise RuntimeError(
            f"Unexpected MNIST output: {scores.shape}"
        )

    # Модель отдаёт логиты. argmax можно брать напрямую,
    # softmax используем только для confidence.
    probs = np.exp(scores - scores.max())
    probs /= probs.sum()

    digit = int(probs.argmax())

    return (
        str(digit),
        round(float(probs[digit]), 3),
        prepared,
    )


def make_soft_ink(
    gray: np.ndarray,
    gamma: float = 1.5,
    low: float = 8.0,
    high: float = 125.0,
) -> np.ndarray:
    """
    Преобразует grayscale в мягкую маску чернил.

    Выход:
        0   = фон
        255 = самый тёмный штрих

    Сохраняет промежуточные оттенки.
    """

    image = gray.astype(np.float32)

    paper = float(np.percentile(image, 90))

    darkness = np.maximum(
        paper - image,
        0.0,
    )

    strength = np.clip(
        (darkness - low) / (high - low),
        0.0,
        1.0,
    )

    strength = np.power(strength, gamma)

    return np.round(
        strength * 255
    ).astype(np.uint8)


def _mask_and_digits(
    gray: np.ndarray,
    cx: int,
    cy: int,
    pitch: float,
):
    ink = make_soft_ink(
        gray,
        gamma=1.5,
        low=8.0,
        high=125.0,
    )

    h, w = ink.shape

    strong = ink >= 90

    rows = np.ones(h, dtype=bool)

    band_y = max(2, round(pitch * 0.045))

    rows[
        max(0, cy - band_y):
        min(h, cy + band_y + 1)
    ] = False

    if not np.any(rows):
        return ink, [], "invalid_roi"

    profile = np.count_nonzero(
        strong[rows, :],
        axis=0,
    ).astype(np.float32)

    band_x = max(1, round(pitch * 0.025))

    for x in range(
        max(0, cx - band_x),
        min(w, cx + band_x + 1),
    ):
        if profile[x] > rows.sum() * 0.75:
            profile[x] = 0

    profile = np.convolve(
        profile,
        [0.25, 0.5, 0.25],
        mode="same",
    )

    min_signal = max(
        2.0,
        rows.sum() * 0.07,
    )

    occupied = profile >= min_signal

    padded = np.pad(
        occupied.astype(np.int8),
        (1, 1),
    )

    transitions = np.diff(padded)

    starts = np.where(
        transitions == 1
    )[0]

    ends = np.where(
        transitions == -1
    )[0]

    raw_runs = [
        (int(a), int(b))
        for a, b in zip(starts, ends)
    ]

    if not raw_runs:
        return ink, [], "blank_or_uncertain"

    max_gap = max(
        2,
        round(pitch * 0.06),
    )

    merged = []

    for a, b in raw_runs:
        if not merged:
            merged.append([a, b])
            continue

        previous = merged[-1]

        gap = a - previous[1]

        if gap <= max_gap:
            previous[1] = b
        else:
            merged.append([a, b])

    min_width = max(
        2,
        round(pitch * 0.025),
    )

    max_profile = float(profile.max())

    runs = []

    for a, b in merged:
        width = b - a
        strength = float(profile[a:b].sum())

        if width < min_width:
            continue

        if strength < max_profile * 0.20:
            continue

        runs.append((a, b))

    if not runs:
        return ink, [], "no_segments"

    if len(runs) > 3:
        widths = [b - a for a, b in runs]

        return (
            ink,
            [],
            f"uncertain_split:{len(runs)} "
            f"widths={widths}",
        )

    digits = []

    for a, b in runs:
        margin = max(1, round(pitch * 0.02))

        left = max(0, a - margin)
        right = min(w, b + margin)

        fragment = ink[:, left:right].copy()

        digits.append(fragment)

    return ink, digits, "ok"


def _crnn_text(gray: np.ndarray) -> str:
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

        roi = gray[y1:y2, x1:x2].copy()

        cleaned, digits, status = _mask_and_digits(
            roi,
            cx - x1,
            cy - y1,
            pitch,
        )

        roi_h, roi_w = cleaned.shape

        yy, xx = np.ogrid[:roi_h, :roi_w]

        local_cx = cx - x1
        local_cy = cy - y1

        band = max(1, round(pitch * 0.04))

        outside_grid = (
            (np.abs(xx - local_cx) > band)
            &
            (np.abs(yy - local_cy) > band)
        )

        pixels = cleaned[outside_grid]

        ink_score = (
            np.count_nonzero(pixels) / pixels.size
            if pixels.size else 0.0
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

    if cells is None:
        proposals.sort(
            reverse=True,
            key=lambda item: item[0],
        )
        proposals = proposals[:max_rois]

    comparisons = []

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
                        _png_b64(mnist_img)
                    )

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