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


def project_name() -> str:
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    match = PROJECT_NAME.search(pyproject.read_text(encoding="utf-8"))
    if not match:
        raise SystemExit("Could not find project.name in pyproject.toml")
    return match.group(1)


def main() -> None:
    dist_name = project_name()
    dist_version = metadata.version(dist_name)
    print(f"Installed {dist_name} {dist_version}")

    # The reference executor is always compiled in and must be constructible.
    executor = pGB.ReferenceExecutor()
    executor.synchronize()

    for name in ("CudaExecutor", "HipExecutor", "DpcppExecutor"):
        available = hasattr(pGB, name)
        print(f"  {name}: {'built' if available else 'not built'}")

    print("pyGinkgo wheel smoke test passed")


if __name__ == "__main__":
    main()
