#pragma once

#include <cstdint>
#include <vector>

namespace kifu {

// Реверс пикселей RGBA: первый пиксель становится последним, и наоборот.
// bytes.size() должно делиться на 4. Возвращает новый буфер.
std::vector<std::uint8_t> reverse_pixels(const std::vector<std::uint8_t>& rgba);

}  // namespace kifu