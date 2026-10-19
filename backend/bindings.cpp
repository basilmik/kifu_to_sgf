#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "kifu.hpp"

namespace py = pybind11;

py::bytes reverse(py::bytes data) {
    std::string buf = data;
    std::vector<std::uint8_t> in(buf.begin(), buf.end());
    auto out = kifu::reverse_pixels(in);
    return py::bytes(reinterpret_cast<const char*>(out.data()), out.size());
}

py::dict detect_board(py::bytes data, int width, int height) {
    std::string buf = data;
    std::vector<std::uint8_t> in(buf.begin(), buf.end());

    auto r = kifu::detect_board(in, width, height);

    py::dict d;
    d["found"] = r.found;
    d["marked_rgba"] = py::bytes(
        reinterpret_cast<const char*>(r.marked_rgba.data()),
        r.marked_rgba.size());
    py::list corners;
    for (const auto& p : r.corners) {
        corners.append(py::make_tuple(p.first, p.second));
    }
    d["corners"] = corners;
    return d;
}

PYBIND11_MODULE(_kifu_core, m) {
    m.doc() = "kifu core bindings";
    m.def("reverse", &reverse, "Reverse RGBA pixels");
    m.def("detect_board", &detect_board, "Detect board rectangle on RGBA image");
}