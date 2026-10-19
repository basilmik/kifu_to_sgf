#include "kifu.hpp"

#include <stdexcept>

namespace kifu {

std::vector<std::uint8_t> reverse_pixels(const std::vector<std::uint8_t>& rgba) {
    if (rgba.size() % 4 != 0) {
        throw std::invalid_argument("RGBA buffer size must be a multiple of 4");
    }
    std::vector<std::uint8_t> out(rgba.size());
    const std::size_t n = rgba.size() / 4;  // число пикселей
    for (std::size_t i = 0; i < n; ++i) {
        const std::size_t src = (n - 1 - i) * 4;
        const std::size_t dst = i * 4;
        out[dst + 0] = rgba[src + 0];
        out[dst + 1] = rgba[src + 1];
        out[dst + 2] = rgba[src + 2];
        out[dst + 3] = rgba[src + 3];
    }
    return out;
}

}  // namespace kifu