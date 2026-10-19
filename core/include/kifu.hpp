#pragma once

#include <cstdint>
#include <utility>
#include <vector>

namespace kifu {

// Реверс RGBA-пикселей (уже было)
std::vector<std::uint8_t> reverse_pixels(const std::vector<std::uint8_t>& rgba);

struct DetectResult {
    bool found = false;
    // 4 угла в порядке TL, TR, BR, BL (в пикселях исходного фото)
    std::vector<std::pair<int, int>> corners;
    // RGBA-изображение с нарисованным контуром (тот же размер, что вход)
    std::vector<std::uint8_t> marked_rgba;
};

// Ищет самый крупный четырёхугольник (доску) на фото и рисует его.
// Вход: RGBA, ширина, высота (в пикселях, RGBA-порядок как у Pillow).
DetectResult detect_board(const std::vector<std::uint8_t>& rgba,
                          int width,
                          int height);

}  // namespace kifu