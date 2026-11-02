import re
from collections import defaultdict

import cv2
import numpy as np
import pytesseract
import base64
import threading

from functools import lru_cache
from PIL import Image
from pytesseract import Output

_paddle_lock = threading.RLock()


@lru_cache(maxsize=1)
def get_paddle_model():
    from paddleocr import TextRecognition

    return TextRecognition(
        model_name="PP-OCRv5_server_rec",
        device="cpu",
        cpu_threads=2,
    )


def paddle_recognize_batch(crops):
    # Сериализуем доступ к модели из разных
    # FastAPI-запросов.
    with _paddle_lock:
        model = get_paddle_model()

        results = model.predict(
            input=crops,
            batch_size=8,
        )

        output = []

        for result in results:
            data = result.json["res"]

            output.append({
                "text": str(data.get("rec_text", "")),
                "confidence": round(
                    float(data.get("rec_score", 0)) * 100,
                    1,
                ),
            })

        return output

def tesseract_recognize(crop):
    data = pytesseract.image_to_data(
        crop,
        config=(
            "--oem 1 --psm 7 "
            "-c tessedit_char_whitelist=0123456789"
        ),
        output_type=Output.DICT,
        timeout=4,
    )

    words = []
    scores = []

    for text, confidence in zip(
        data["text"],
        data["conf"],
    ):
        text = str(text).strip()
        confidence = float(confidence)

        if text and confidence >= 0:
            words.append(text)
            scores.append(confidence)

    return {
        "text": "".join(words),
        "confidence": (
            round(min(scores), 1)
            if scores else 0.0
        ),
    }

def compare_ocr(
    rectified: Image.Image,
    x_lines: list[float],
    y_lines: list[float],
    max_rois: int = 100,
) -> dict:
    grid_size = len(x_lines)

    if grid_size not in (9, 13, 19):
        raise ValueError("unsupported grid size")

    if len(y_lines) != grid_size:
        raise ValueError("invalid grid lines")

    if not 1 <= max_rois <= grid_size * grid_size:
        raise ValueError("invalid max_rois")

    gray = cv2.cvtColor(
        np.asarray(rectified.convert("RGB")),
        cv2.COLOR_RGB2GRAY,
    )

    h, w = gray.shape

    dx = float(np.median(np.diff(x_lines)))
    dy = float(np.median(np.diff(y_lines)))

    if dx <= 0 or dy <= 0:
        raise ValueError("invalid grid spacing")

    pitch = min(dx, dy)
    radius = max(8, round(pitch * 0.48))

    # Маска используется только для определения
    # интересных областей, но НЕ передаётся в OCR.
    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        12,
    )

    kernel_len = max(9, round(pitch * 0.75))

    vertical = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (1, kernel_len),
        ),
    )

    horizontal = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (kernel_len, 1),
        ),
    )

    mask = cv2.subtract(
        binary,
        cv2.bitwise_or(vertical, horizontal),
    )

    proposals = []

    # 1. Собираем области вокруг всех пересечений.
    for row, y in enumerate(y_lines):
        for col, x in enumerate(x_lines):
            cx, cy = round(x), round(y)

            x1 = max(0, cx - radius)
            y1 = max(0, cy - radius)

            x2 = min(w, cx + radius + 1)
            y2 = min(h, cy + radius + 1)

            roi = gray[y1:y2, x1:x2].copy()
            roi_mask = mask[y1:y2, x1:x2]

            if roi.size == 0:
                continue

            # Сколько пикселей осталось после
            # предполагаемого удаления сетки.
            ink_score = (
                cv2.countNonZero(roi_mask)
                / roi_mask.size
            )

            # Для ОБЕИХ моделей одинаково увеличиваем
            # исходный ROI и добавляем белое поле.
            # Штрихи сетки при этом не удаляются.
            crop = cv2.resize(
                roi,
                None,
                fx=3,
                fy=3,
                interpolation=cv2.INTER_CUBIC,
            )

            crop = cv2.copyMakeBorder(
                crop,
                15, 15, 15, 15,
                cv2.BORDER_CONSTANT,
                value=255,
            )

            proposals.append({
                "row": row,
                "col": col,
                "ink_score": ink_score,
                "crop": crop,
            })

    # 2. Для первого эксперимента ограничиваем
    # число ROI, иначе Tesseract может работать долго.
    #
    # Отбираем области с наибольшим остатком чернил.
    proposals.sort(
        key=lambda p: p["ink_score"],
        reverse=True,
    )

    selected = proposals[:max_rois]

    if not selected:
        return {
            "comparisons": [],
            "inspected_cells": grid_size * grid_size,
            "ocr_attempts": 0,
        }

    # 3. PaddleOCR получает те же исходные ROI,
    # что и Tesseract.
    # Преобразование grayscale -> BGR меняет
    # формат каналов, но не содержимое картинки.
    paddle_crops = [
        cv2.cvtColor(
            p["crop"],
            cv2.COLOR_GRAY2BGR,
        )
        for p in selected
    ]

    paddle_results = paddle_recognize_batch(
        paddle_crops
    )

    if len(paddle_results) != len(selected):
        raise RuntimeError(
            "PaddleOCR result count mismatch"
        )

    # 4. Tesseract запускается на этих же фрагментах.
    comparisons = []

    for item, paddle_result in zip(
        selected,
        paddle_results,
    ):
        crop = item["crop"]

        tesseract_result = tesseract_recognize(crop)

        # Отдаём сам ROI на фронтенд для диагностики.
        ok, png = cv2.imencode(".png", crop)

        if not ok:
            raise RuntimeError("cannot encode ROI")

        comparisons.append({
            "row": item["row"],
            "col": item["col"],
            "ink_score": round(
                item["ink_score"], 4
            ),
            "roi_png_b64": base64.b64encode(
                png.tobytes()
            ).decode("ascii"),
            "tesseract": tesseract_result,
            "paddle": paddle_result,
        })

    comparisons.sort(
        key=lambda item: (
            item["row"],
            item["col"],
        )
    )

    return {
        "comparisons": comparisons,
        "inspected_cells": grid_size * grid_size,
        "ocr_attempts": len(comparisons),
    }

def recognize_numbers(
    rectified: Image.Image,
    x_lines: list[float],
    y_lines: list[float],
    max_move: int = 500,
) -> dict:
    grid_size = len(x_lines)

    if grid_size not in (9, 13, 19):
        raise ValueError("unsupported grid size")

    if len(y_lines) != grid_size:
        raise ValueError("invalid grid line count")

    w, h = rectified.size

    if not (
        all(np.isfinite(x_lines))
        and all(np.isfinite(y_lines))
    ):
        raise ValueError("non-finite coordinates")

    if (
        any(x < 0 or x >= w for x in x_lines)
        or any(y < 0 or y >= h for y in y_lines)
    ):
        raise ValueError("grid outside image")

    if (
        any(a >= b for a, b in zip(x_lines, x_lines[1:]))
        or any(a >= b for a, b in zip(y_lines, y_lines[1:]))
    ):
        raise ValueError("grid lines must be increasing")

    if not 1 <= max_move <= 2000:
        raise ValueError("invalid max_move")

    # 1. Grayscale
    gray = cv2.cvtColor(
        np.asarray(rectified.convert("RGB")),
        cv2.COLOR_RGB2GRAY
    )

    # 2. Бинаризация: тёмные элементы становятся белыми.
    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        12
    )

    # 3. Средний шаг сетки.
    dx = float(np.median(np.diff(x_lines)))
    dy = float(np.median(np.diff(y_lines)))

    pitch = min(dx, dy)

    # ROI немного меньше расстояния между пересечениями.
    radius = max(6, int(pitch * 0.43))

    # 4. Выделяем длинные линии сетки.
    vertical_length = max(9, round(dy * 0.75))
    horizontal_length = max(9, round(dx * 0.75))

    vertical_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (1, vertical_length)
    )

    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (horizontal_length, 1)
    )

    vertical = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        vertical_kernel
    )

    horizontal = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        horizontal_kernel
    )

    # 5. Убираем обнаруженные линии сетки.
    # grid_mask = cv2.bitwise_or(vertical, horizontal)

    # cleaned = cv2.subtract(binary, grid_mask)

    candidates = []
    ocr_attempts = 0

    # 6. Перебираем пересечения.
    for row, y in enumerate(y_lines):
        for col, x in enumerate(x_lines):

            cx = round(x)
            cy = round(y)

            x1 = max(0, cx - radius)
            y1 = max(0, cy - radius)

            x2 = min(w, cx + radius + 1)
            y2 = min(h, cy + radius + 1)

            # roi = cleaned[y1:y2, x1:x2]
            roi_mask = cleaned[y1:y2, x1:x2]

            min_ink = max(16, int(roi_mask.size * 0.012))

            if cv2.countNonZero(roi_mask) < min_ink:
                continue

            ocr_attempts += 1

            # Но в OCR отправляем исходное изображение,
            # чтобы сохранить все штрихи цифр.
            roi_original = gray[y1:y2, x1:x2]

            crop = cv2.resize(
                roi_original,
                None,
                fx=3,
                fy=3,
                interpolation=cv2.INTER_CUBIC
            )

            crop = cv2.copyMakeBorder(
                crop,
                15, 15, 15, 15,
                cv2.BORDER_CONSTANT,
                value=255
            )

            # Пропускаем почти пустые фрагменты.
            # min_ink = max(
            #     16,
            #     int(roi.size * 0.012)
            # )

            # if cv2.countNonZero(roi) < min_ink:
            #     continue

            # ocr_attempts += 1

            # # 7. Подготавливаем фрагмент для Tesseract.
            # # Tesseract получает тёмный текст на белом фоне.
            # crop = 255 - roi

            # crop = cv2.resize(
            #     crop,
            #     None,
            #     fx=3,
            #     fy=3,
            #     interpolation=cv2.INTER_CUBIC
            # )

            # crop = cv2.copyMakeBorder(
            #     crop,
            #     15, 15, 15, 15,
            #     cv2.BORDER_CONSTANT,
            #     value=255
            # )

            # 8. OCR.
            data = pytesseract.image_to_data(
                crop,
                config=(
                    "--oem 1 --psm 7 "
                    "-c tessedit_char_whitelist=0123456789"
                ),
                output_type=Output.DICT,
                timeout=3
            )

            words = [
                (str(text).strip(), float(conf))
                for text, conf in zip(
                    data["text"],
                    data["conf"]
                )
                if str(text).strip()
            ]

            if not words:
                continue

            recognized_text = "".join(
                text for text, _ in words
            )

            if not re.fullmatch(
                r"[0-9]{1,4}",
                recognized_text
            ):
                continue

            number = int(recognized_text)

            confidence = min(
                conf for _, conf in words
            )

            # 9. Отсекаем явно сомнительные результаты.
            if number < 1 or number > max_move:
                continue

            if confidence < 40:
                continue

            candidates.append({
                "number": number,
                "row": row,
                "col": col,
                "confidence": round(confidence, 1)
            })

    candidates.sort(
        key=lambda item: (
            item["number"],
            -item["confidence"]
        )
    )

    result = {
        "candidates": candidates,
        "inspected_cells": grid_size * grid_size,
        "ocr_attempts": ocr_attempts,
    }

    result.update(build_sgf(candidates, grid_size))

    return result

def build_sgf(
    candidates: list[dict],
    grid_size: int
) -> dict:
    grouped = defaultdict(list)

    # Группируем результаты по номеру хода.
    for move in candidates:
        grouped[move["number"]].append(move)

    warnings = []

    duplicates = sorted(
        number
        for number, matches in grouped.items()
        if len(matches) > 1
    )

    if duplicates:
        warnings.append(
            f"Duplicate move numbers: {duplicates[:25]}"
        )

    # Берём только непрерывную последовательность:
    # 1, 2, 3, ...
    sequence = []

    for number in range(
        1,
        max(grouped, default=0) + 1
    ):
        matches = grouped.get(number, [])

        if len(matches) != 1:
            warnings.append(
                f"Stop at move {number}: "
                "missing or ambiguous. "
                "Later moves not exported."
            )
            break

        sequence.append(matches[0])

    if not sequence:
        return {
            "exported_moves": [],
            "sgf": None,
            "warnings": warnings or [
                "Move 1 not recognized."
            ],
        }

    # SGF: a = первая строка/столбец.
    letters = "abcdefghijklmnopqrstuvwxyz"

    nodes = [
        (
            "(;FF[4]GM[1]CA[UTF-8]"
            "AP[KifuOCR:baseline]"
            f"SZ[{grid_size}]"
        )
    ]

    for move in sequence:
        number = move["number"]

        color = "B" if number % 2 else "W"

        col = move["col"]
        row = move["row"]

        coordinate = letters[col] + letters[row]

        nodes.append(
            f";{color}[{coordinate}]"
        )

    nodes.append(")")

    sgf = "\n".join(nodes) + "\n"

    return {
        "exported_moves": sequence,
        "sgf": sgf,
        "warnings": warnings,
    }