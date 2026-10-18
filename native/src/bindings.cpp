// Python bindings for the native power sampler.
//
// Kept as the only file that knows about Python: everything else under
// native/ is plain C++ with its own tests, so the core can be built and
// checked without an interpreter in sight.

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace {

const char* compiler() {
#if defined(__clang__)
    return "clang " __clang_version__;
#elif defined(__GNUC__)
    return "gcc " __VERSION__;
#else
    return "unknown";
#endif
}

}  // namespace

PYBIND11_MODULE(nvml_sampler, m) {
    m.doc() = "High-rate NVML power sampling, outside the GIL.";
    m.attr("__version__") = NVML_SAMPLER_VERSION;

    m.def(
        "build_info",
        [] {
            py::dict info;
            info["version"] = NVML_SAMPLER_VERSION;
            info["compiler"] = compiler();
            info["cxx_standard"] = __cplusplus;
            info["build_type"] = NVML_SAMPLER_BUILD_TYPE;
            return info;
        },
        "How this module was built — recorded in run reports.");
}
