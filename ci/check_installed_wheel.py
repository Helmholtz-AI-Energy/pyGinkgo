#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyGinkgo authors
#
# SPDX-License-Identifier: MIT

"""Smoke test for an already installed pyGinkgo wheel.

cibuildwheel runs this against every repaired CPU wheel. It verifies that the
package imports, that its distribution version agrees with pyproject.toml, and
that the reference executor is usable. GPU backends are reported but not
exercised, since GitHub-hosted runners have no GPU.
"""

from __future__ import annotations

from importlib import metadata
from importlib import util as _importlib_util
from pathlib import Path
import re
import sys


def _report_windows_load_failure(error: BaseException) -> None:
    """Name the library Windows could not resolve.

    "DLL load failed ... The specified module could not be found" never says
    which module is missing, which makes a CI-only failure very hard to act on.
    Loading each bundled binary explicitly by absolute path pinpoints the one
    that fails, and its dependency chain along with it.
    """
    import ctypes

    print(f"Import failed: {error}", file=sys.stderr)

    spec = _importlib_util.find_spec("pyGinkgo")
    if spec is None or not spec.submodule_search_locations:
        print("Could not locate the installed pyGinkgo package.", file=sys.stderr)
        return

    package_dir = Path(list(spec.submodule_search_locations)[0])
    print(f"Package directory: {package_dir}", file=sys.stderr)
    for entry in sorted(package_dir.iterdir()):
        print(f"    {entry.name}", file=sys.stderr)

    print("Loading each bundled binary by absolute path:", file=sys.stderr)
    for binary in sorted(package_dir.glob("*.dll")) + sorted(package_dir.glob("*.pyd")):
        try:
            ctypes.WinDLL(str(binary))
        except OSError as exc:
            print(f"    FAIL {binary.name}: {exc}", file=sys.stderr)
        else:
            print(f"    ok   {binary.name}", file=sys.stderr)


try:
    import pyGinkgo
    import pyGinkgo.pyGinkgoBindings as pGB
except ImportError as error:
    if sys.platform == "win32":
        _report_windows_load_failure(error)
    raise


PROJECT_NAME = re.compile(r'(?m)^name = "([^"]+)"$')
PROJECT_VERSION = re.compile(r'(?m)^version = "([^"]+)"$')


def project_field(pattern: re.Pattern[str], field: str) -> str:
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    match = pattern.search(pyproject.read_text(encoding="utf-8"))
    if not match:
        raise SystemExit(f"Could not find project.{field} in pyproject.toml")
    return match.group(1)


def main() -> None:
    dist_name = project_field(PROJECT_NAME, "name")
    dist_version = metadata.version(dist_name)
    print(f"Installed {dist_name} {dist_version}")

    # Catches a wheel built before scripts/set_package_version.py stamped the
    # version, which would otherwise be published as 0.0.0.
    expected_version = project_field(PROJECT_VERSION, "version")
    if dist_version != expected_version:
        raise SystemExit(
            f"Installed {dist_name} version ({dist_version}) does not match "
            f"the version in pyproject.toml ({expected_version})"
        )

    # The reference executor is always compiled in and must be constructible.
    executor = pGB.ReferenceExecutor()
    executor.synchronize()

    for name in ("CudaExecutor", "HipExecutor", "DpcppExecutor"):
        available = hasattr(pGB, name)
        print(f"  {name}: {'built' if available else 'not built'}")

    print("pyGinkgo wheel smoke test passed")


if __name__ == "__main__":
    main()
