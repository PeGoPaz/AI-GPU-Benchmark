"""The native module itself. Skipped unless it was built (`pip install ./native`).

Everything else in the suite runs without the extension; these tests are the
ones that need it, and CI runs them in the job that builds it.
"""
import pytest

nvml_sampler = pytest.importorskip("nvml_sampler")


def test_version_comes_from_cmake():
    assert nvml_sampler.__version__.count(".") == 2


def test_build_info_names_the_toolchain():
    info = nvml_sampler.build_info()

    assert info["version"] == nvml_sampler.__version__
    assert info["cxx_standard"] >= 201703
    assert info["compiler"]
