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
    # Мягкий контраст, без бинаризации Otsu.
    # Все оттенки сохраняются.
    ink = make_soft_ink(
        gray,
        gamma=1.5,
        low=8.0,
        high=125.0,
    )

    h, w = ink.shape

    # Бинарное представление используем
    # ТОЛЬКО для анализа положения штрихов.
    strong = ink >= 90

    # 1. Не учитываем центральную горизонтальную
    # линию при построении вертикальной проекции.
    rows = np.ones(h, dtype=bool)

    band_y = max(2, round(pitch * 0.045))

    rows[
        max(0, cy - band_y):
        min(h, cy + band_y + 1)
    ] = False

    if not np.any(rows):
        return ink, [], "invalid_roi"

    # 2. Вертикальная проекция.
    profile = np.count_nonzero(
        strong[rows, :],
        axis=0,
    ).astype(np.float32)

    # 3. Подавляем предполагаемую вертикальную
    # линию сетки только в проекции.
    #
    # Исходное изображение ink не изменяется!
    band_x = max(1, round(pitch * 0.025))

    for x in range(
        max(0, cx - band_x),
        min(w, cx + band_x + 1),
    ):
        if profile[x] > rows.sum() * 0.75:
            profile[x] = 0

    # 4. Лёгкое сглаживание проекции.
    profile = np.convolve(
        profile,
        [0.25, 0.5, 0.25],
        mode="same",
    )

    # 5. Находим столбцы с достаточно
    # выраженными тёмными штрихами.
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

    # 6. Объединяем близкие группы.
    #
    # Небольшие разрывы могут возникать
    # внутри одной и той же цифры.
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

    # 7. Фильтруем слабые изолированные группы.
    #
    # Не удаляем их из изображения — только
    # исключаем из кандидатов сегментации.
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

    # 8. Проверяем количество групп.
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

    # 9. Вырезаем фрагменты из исходной
    # мягкой маски, без удаления линий.
    digits = []

    for a, b in runs:
        # Небольшой запас по горизонтали.
        margin = max(1, round(pitch * 0.02))

        left = max(0, a - margin)
        right = min(w, b + margin)

        fragment = ink[:, left:right].copy()

        digits.append(fragment)

    return ink, digits, "ok"


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

    # Яркость бумаги оцениваем по светлым
    # пикселям самого ROI.
    paper = float(np.percentile(image, 90))

    # Насколько каждый пиксель темнее бумаги.
    darkness = np.maximum(
        paper - image,
        0.0,
    )

    # Небольшое подавление слабых оттенков.
    strength = np.clip(
        (darkness - low) / (high - low),
        0.0,
        1.0,
    )

    # Усиливаем различие между светлыми
    # линиями и тёмными цифрами.
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
    # 1. Бинаризация без удаления линий.
    # Белые пиксели = тёмные элементы изображения.
    # _, mask = cv2.threshold(
    #     gray,
    #     0,
    #     255,
    #     cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU,
    # )

    mask = make_soft_ink(
        gray,
        gamma=1.5,
        low=8.0,
        high=125.0,
    )

    h, w = mask.shape

    # 2. Вертикальная проекция.
    #
    # Центральную горизонтальную линию не учитываем
    # только при расчёте проекции.
    # В самом изображении она остаётся!

    band = max(1, round(pitch * 0.035))

    projection_rows = np.ones(h, dtype=bool)

    projection_rows[
        max(0, cy - band):
        min(h, cy + band + 1)
    ] = False

    if not projection_rows.any():
        return mask, [], "invalid_roi"

    # projection = np.count_nonzero(
    #     mask[projection_rows, :],
    #     axis=0,
    # )

    projection = np.count_nonzero(
        mask[projection_rows, :] >= 90,
        axis=0,
    )

    # 3. Определяем столбцы со штрихами.
    min_signal = max(
        2,
        round(projection_rows.sum() * 0.05),
    )

    occupied = projection >= min_signal

    # 4. Находим непрерывные группы столбцов.
    padded = np.pad(
        occupied.astype(np.int8),
        (1, 1),
    )

    transitions = np.diff(padded)

    starts = np.where(transitions == 1)[0]
    ends = np.where(transitions == -1)[0]

    min_width = max(
        2,
        round(pitch * 0.035),
    )

    runs = [
        (int(a), int(b))
        for a, b in zip(starts, ends)
        if b - a >= min_width
    ]

    # 5. Проверяем, получилось ли разделение.
    if len(runs) == 0:
        return mask, [], "blank_or_uncertain"

    if len(runs) > 3:
        return mask, [], f"uncertain_split:{len(runs)}"

    # Очень широкая единая группа может означать,
    # что несколько цифр касаются друг друга.
    if (
        len(runs) == 1
        and runs[0][1] - runs[0][0] > pitch * 0.5
    ):
        return mask, [], "possibly_joined_digits"

    # 6. Вырезаем предполагаемые цифры.
    #
    # ВАЖНО: используем исходную бинарную маску.
    # Никаких линий сетки из неё не удаляли.
    digits = []

    for a, b in runs:
        fragment = mask[:, a:b].copy()

        digits.append(fragment)

    return mask, digits, "ok"

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

def _mnist_digit(mask: np.ndarray):
    h, w = mask.shape

    if h == 0 or w == 0:
        raise ValueError("Empty digit image")

    # Масштабируем цифру, сохраняя пропорции
    scale = 20.0 / max(h, w)

    new_w = max(1, round(w * scale))
    new_h = max(1, round(h * scale))

    scaled = cv2.resize(
        mask,
        (new_w, new_h),
        interpolation=(
            cv2.INTER_AREA
            if scale < 1
            else cv2.INTER_CUBIC
        ),
    )

    # MNIST: 28x28, белая цифра на чёрном фоне
    img = np.zeros(
        (28, 28),
        dtype=np.uint8,
    )

    y = (28 - new_h) // 2
    x = (28 - new_w) // 2

    img[
        y:y + new_h,
        x:x + new_w,
    ] = scaled

    # Запускаем ONNX-модель
    session = _mnist_session()

    input_name = session.get_inputs()[0].name

    tensor = (
        img.astype(np.float32) / 255.0
    )[None, None, ...]

    scores = session.run(
        None,
        {input_name: tensor},
    )[0].reshape(-1)

    if scores.size != 10:
        raise RuntimeError(
            f"Unexpected MNIST output: {scores.shape}"
        )

    # Преобразуем оценки классов в softmax
    probs = np.exp(scores - scores.max())
    probs /= probs.sum()

    digit = int(probs.argmax())

    return (
        str(digit),
        round(float(probs[digit]), 3),
        img,
    )

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

        # Оцениваем количество штрихов ВНЕ центральных
        # полос сетки, не изменяя само изображение.

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