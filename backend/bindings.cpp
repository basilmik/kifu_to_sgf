#include <pybind11/pybind11.h>
#include "kifu.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_kifu_core, m) {
    m.doc() = "kifu core bindings";
    m.def("add", &kifu::add, "Add two integers");
}