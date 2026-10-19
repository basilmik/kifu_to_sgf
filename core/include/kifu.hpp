#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace kifu {

std::vector<std::uint8_t> reverse_pixels(const std::vector<std::uint8_t>& rgba);

struct DetectResult {
    bool found = false;
    std::vector<std::pair<int, int>> corners;
    std::vector<std::uint8_t> marked_rgba;
};

// Отладочная информация: одна запись на шаг конвейера
struct DebugImage {
    std::string name;                 // "edges", "contours", "marked"
    int width = 0;
    int height = 0;
    std::vector<std::uint8_t> rgba;   // всегда RGBA, чтобы Python/Pillow не конвертировал
};

struct ContourInfo {
    double area = 0;
    double perimeter = 0;
    int approx_points = 0;            // сколько точек после approxPolyDP
    bool reasonable = false;          // прошёл ли is_reasonable_quad
    std::vector<std::pair<int, int>> points; // сам аппроксимированный контур (до 4 точек)
};

struct DebugResult {
    DetectResult detect;
    std::vector<DebugImage> images;   // grayscale, edges, dilated, marked
    std::vector<ContourInfo> top_contours;  // топ-10 по площади
    double img_area = 0;
    std::string note;                 // краткое пояснение, почему не нашли
};

DetectResult detect_board(const std::vector<std::uint8_t>& rgba, int width, int height);

// Отладочная версия того же детектора
DebugResult detect_board_debug(const std::vector<std::uint8_t>& rgba, int width, int height);

}  // namespace kifu