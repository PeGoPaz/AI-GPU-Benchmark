"""The installable package: one version, one dependency list, one entry point."""
import re
import subprocess
import sys
from pathlib import Path

import pytest

import gpu_bench
from gpu_bench.cli import build_parser

ROOT = Path(__file__).resolve().parent.parent


def _pyproject() -> dict:
    tomllib = pytest.importorskip("tomllib")  # 3.11+; the 3.10 job skips these
    with open(ROOT / "pyproject.toml", "rb") as f:
        return tomllib.load(f)


def test_version_is_pep_440():
    assert re.fullmatch(r"\d+\.\d+\.\d+((a|b|rc)\d+)?(\.post\d+)?(\.dev\d+)?",
                        gpu_bench.__version__)


def test_version_and_dependencies_come_from_one_place():
    project = _pyproject()
    dynamic = project["tool"]["setuptools"]["dynamic"]

    assert set(project["project"]["dynamic"]) == {"version", "dependencies"}
    assert dynamic["version"] == {"attr": "gpu_bench.__version__"}
    assert dynamic["dependencies"] == {"file": ["requirements.txt"]}


def test_console_script_points_at_main():
    scripts = _pyproject()["project"]["scripts"]

    module, _, func = scripts["gpu-bench"].partition(":")
    assert (module, func) == ("gpu_bench.__main__", "main")


def test_the_icon_ships_as_package_data():
    from gpu_bench.ui import about

    patterns = _pyproject()["tool"]["setuptools"]["package-data"]["gpu_bench.ui"]
    icon = Path(about.ICON_PATH).relative_to(ROOT / "gpu_bench" / "ui")
    assert any(icon.match(pattern) for pattern in patterns)


def test_version_flag(capsys):
    with pytest.raises(SystemExit) as exit_info:
        build_parser().parse_args(["--version"])

    assert exit_info.value.code == 0
    assert capsys.readouterr().out.strip() == f"gpu-bench {gpu_bench.__version__}"


def test_runs_as_a_module():
    result = subprocess.run([sys.executable, "-m", "gpu_bench", "--list"],
                            capture_output=True, text=True, cwd=ROOT, timeout=120)

    assert result.returncode == 0, result.stderr
    assert "Presets:" in result.stdout
