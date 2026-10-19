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
    for (const auto& p : r.corners) corners.append(py::make_tuple(p.first, p.second));
    d["corners"] = corners;
    return d;
}

py::dict detect_board_debug(py::bytes data, int width, int height) {
    std::string buf = data;
    std::vector<std::uint8_t> in(buf.begin(), buf.end());
    auto r = kifu::detect_board_debug(in, width, height);

    py::dict d;
    d["found"] = r.detect.found;
    d["note"] = r.note;
    d["img_area"] = r.img_area;

    py::list corners;
    for (const auto& p : r.detect.corners) corners.append(py::make_tuple(p.first, p.second));
    d["corners"] = corners;

    d["marked_rgba"] = py::bytes(
        reinterpret_cast<const char*>(r.detect.marked_rgba.data()),
        r.detect.marked_rgba.size());

    py::list images;
    for (const auto& img : r.images) {
        py::dict di;
        di["name"] = img.name;
        di["width"] = img.width;
        di["height"] = img.height;
        di["rgba"] = py::bytes(
            reinterpret_cast<const char*>(img.rgba.data()), img.rgba.size());
        images.append(di);
    }
    d["images"] = images;

    py::list contours;
    for (const auto& ci : r.top_contours) {
        py::dict dc;
        dc["area"] = ci.area;
        dc["perimeter"] = ci.perimeter;
        dc["approx_points"] = ci.approx_points;
        dc["reasonable"] = ci.reasonable;
        py::list pts;
        for (const auto& p : ci.points) pts.append(py::make_tuple(p.first, p.second));
        dc["points"] = pts;
        contours.append(dc);
    }
    d["top_contours"] = contours;

    return d;
}

PYBIND11_MODULE(_kifu_core, m) {
    m.doc() = "kifu core bindings";
    m.def("reverse", &reverse, "Reverse RGBA pixels");
    m.def("detect_board", &detect_board, "Detect board rectangle");
    m.def("detect_board_debug", &detect_board_debug, "Detect board rectangle (debug)");
}