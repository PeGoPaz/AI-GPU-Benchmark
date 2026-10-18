// Python bindings for the native power sampler.
//
// Kept as the only file that knows about Python: everything else under
// native/ is plain C++ with its own tests, so the core can be built and
// checked without an interpreter in sight.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <chrono>
#include <iterator>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include "power/sampler.hpp"

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

std::vector<std::string> library_paths_or_default(
    const std::optional<std::vector<std::string>>& paths) {
    return paths ? *paths : power::nvml::Library::default_paths();
}

// Owns the C++ sampler for Python. Adds what the core leaves to its caller:
// a lock on the consumer side, because Python code — unlike the C++ core's
// single consumer — could call drain() from two threads at once.
class PySampler {
  public:
    PySampler(std::unique_ptr<power::PowerSource> source, long long interval_us,
              std::size_t capacity)
        : sampler_(std::move(source), std::chrono::microseconds{interval_us}, capacity) {}

    void start() { sampler_.start(); }

    void stop() {
        // Joining waits for at most one read; no reason to hold the GIL
        // and stall every other Python thread meanwhile.
        py::gil_scoped_release release;
        sampler_.stop();
    }

    bool running() const { return sampler_.running(); }

    py::tuple drain() {
        std::vector<power::Sample> samples;
        {
            py::gil_scoped_release release;
            std::lock_guard<std::mutex> lock(drain_mutex_);
            sampler_.drain(std::back_inserter(samples));
        }

        const auto n = static_cast<py::ssize_t>(samples.size());
        py::array_t<double> seconds(n);
        py::array_t<float> watts(n);
        auto s = seconds.mutable_unchecked<1>();
        auto w = watts.mutable_unchecked<1>();
        for (py::ssize_t i = 0; i < n; ++i) {
            const auto& sample = samples[static_cast<std::size_t>(i)];
            s(i) = static_cast<double>(sample.unix_ns) / 1e9;
            w(i) = static_cast<float>(sample.milliwatts) / 1000.0f;
        }
        return py::make_tuple(seconds, watts);
    }

    py::dict stats() const {
        const power::SamplerStats st = sampler_.stats();
        py::dict d;
        d["reads"] = st.reads;
        d["errors"] = st.errors;
        d["dropped"] = st.dropped;
        d["changes"] = st.changes;
        d["worst_gap_ms"] = static_cast<double>(st.worst_gap_ns) / 1e6;
        return d;
    }

    std::string description() const { return sampler_.source().description(); }
    long long interval_us() const { return sampler_.interval().count(); }

  private:
    power::Sampler sampler_;
    std::mutex drain_mutex_;
};

std::unique_ptr<PySampler> nvml_sampler(unsigned int device_index, long long interval_us,
                                        std::size_t capacity,
                                        const std::optional<std::vector<std::string>>& paths) {
    auto opened = power::NvmlPowerSource::open(device_index, library_paths_or_default(paths));
    if (!opened.source) {
        throw std::runtime_error(opened.error);
    }
    return std::make_unique<PySampler>(std::move(opened.source), interval_us, capacity);
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

    m.def(
        "available",
        [](unsigned int device_index,
           const std::optional<std::vector<std::string>>& paths) -> py::tuple {
            auto opened =
                power::NvmlPowerSource::open(device_index, library_paths_or_default(paths));
            if (!opened.source) {
                return py::make_tuple(false, opened.error);
            }
            return py::make_tuple(true, opened.source->description());
        },
        py::arg("device_index") = 0, py::arg("library_paths") = py::none(),
        "(True, what will be measured) if a sampler can be opened, else (False, why not).");

    py::class_<PySampler>(m, "PowerSampler",
                          "Reads GPU power on a native thread; drain() collects the samples.")
        .def(py::init(&nvml_sampler), py::arg("device_index") = 0, py::arg("interval_us") = 2000,
             py::arg("capacity") = power::Sampler::kDefaultCapacity,
             py::arg("library_paths") = py::none(),
             "Opens NVML for the given GPU. Raises RuntimeError with the reason if it cannot.")
        .def("start", &PySampler::start)
        .def("stop", &PySampler::stop)
        .def_property_readonly("running", &PySampler::running)
        .def("drain", &PySampler::drain,
             "Everything sampled since the last call, as (unix seconds float64, watts float32).")
        .def("stats", &PySampler::stats)
        .def_property_readonly("description", &PySampler::description)
        .def_property_readonly("interval_us", &PySampler::interval_us)
        .def("__enter__",
             [](PySampler& self) -> PySampler& {
                 self.start();
                 return self;
             })
        .def("__exit__", [](PySampler& self, const py::args&) { self.stop(); });

    auto testing = m.def_submodule("_testing", "Samplers that need no GPU, for tests.");
    testing.def(
        "scripted_sampler",
        [](std::vector<std::optional<std::uint32_t>> readings_mw, long long interval_us,
           long long read_time_us, std::size_t capacity) {
            auto source = std::make_unique<power::ScriptedSource>(
                std::move(readings_mw), std::chrono::microseconds{read_time_us});
            return std::make_unique<PySampler>(std::move(source), interval_us, capacity);
        },
        py::arg("readings_mw"), py::arg("interval_us") = 2000, py::arg("read_time_us") = 0,
        py::arg("capacity") = power::Sampler::kDefaultCapacity,
        "A PowerSampler replaying readings_mw in a loop; None entries are failed reads.");
}
