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
from pathlib import Path
import re

import pyGinkgo
import pyGinkgo.pyGinkgoBindings as pGB


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
