#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "kifu.hpp"

namespace py = pybind11;

py::bytes reverse(py::bytes data) {
    std::string buf = data;  // копия сырых байтов
    std::vector<std::uint8_t> in(buf.begin(), buf.end());
    auto out = kifu::reverse_pixels(in);
    return py::bytes(reinterpret_cast<const char*>(out.data()), out.size());
}

PYBIND11_MODULE(_kifu_core, m) {
    m.doc() = "kifu core bindings";
    m.def("reverse", &reverse, "Reverse RGBA pixels");
}