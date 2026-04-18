# SPDX-FileCopyrightText: 2024 - 2026 pyGinkgo authors
#
# SPDX-License-Identifier: MIT
"""Distributed (MPI) extensions to pyGinkgo.

This subpackage exposes Ginkgo's ``gko::experimental::distributed`` types
to Python: distributed ``Partition``, ``Vector``, and ``Matrix`` classes
plus a matrix-free ``PyLinOp`` trampoline. It is only available when
pyGinkgo was built with ``-DpyGinkgo_BUILD_MPI=ON`` *and* mpi4py is
installed at runtime.

Importing this module:

* raises :class:`ImportError` if the C++ extension was built without MPI;
* raises :class:`ImportError` if mpi4py is missing;
* raises :class:`ImportError` if the runtime mpi4py reports a different
  MPI library/version from the one pyGinkgo was linked against (e.g.
  pyGinkgo built against MPICH but mpi4py built against OpenMPI).

The ABI verification is invoked lazily on first use of any function
accepting a communicator (so that just installing the package is safe
even on non-MPI nodes).
"""

from __future__ import annotations

from .. import pyGinkgoBindings as _pgb

if not hasattr(_pgb, "mpi") or not hasattr(_pgb, "distributed"):
    raise ImportError(
        "pyGinkgo was built without MPI support. Rebuild with "
        "-DpyGinkgo_BUILD_MPI=ON or install the conda variant "
        "`pyginkgo-mpi-{mpich,openmpi}`."
    )

try:
    import mpi4py  # noqa: F401
    from mpi4py import MPI  # noqa: F401
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "pyGinkgo.distributed requires mpi4py. Install with "
        "`pip install mpi4py` or `pip install pyginkgo[mpi]`."
    ) from e


_abi_checked = False


def _ensure_mpi_abi(comm) -> None:
    """Verify the runtime MPI ABI matches what pyGinkgo was built with.

    Called on first use of any distributed entry point that needs a
    communicator. Compares the build-time-baked
    ``MPI_Get_library_version()`` string against the one mpi4py reports
    at runtime, then asks the C++ side to round-trip an MPI call on the
    user-supplied communicator.
    """
    global _abi_checked
    if _abi_checked:
        return
    build_impl = getattr(_pgb.mpi, "BUILD_MPI_IMPL", "unknown")
    build_ver = getattr(_pgb.mpi, "BUILD_MPI_LIBRARY_VERSION", "unknown")
    runtime_ver = MPI.Get_library_version().strip()
    # Heuristic: the impl name (MPICH/Open MPI) should appear in both.
    if build_impl and build_impl.lower() not in runtime_ver.lower() and \
            "unknown" not in build_impl.lower():
        raise ImportError(
            "MPI ABI mismatch: pyGinkgo was built against "
            f"{build_impl!r} ({build_ver!r}) but mpi4py is using a "
            f"different implementation:\n  {runtime_ver}\n"
            "Reinstall a matching pyGinkgo build (e.g. "
            "`conda install pyginkgo-mpi=*=*mpich*`)."
        )
    # C++-side round-trip check
    _pgb.mpi.verify_abi(comm)
    _abi_checked = True


from . import communicator as communicator  # noqa: E402,F401
from . import partition as partition  # noqa: E402,F401
from . import vector as vector  # noqa: E402,F401
from . import matrix as matrix  # noqa: E402,F401
from . import linop as linop  # noqa: E402,F401

# Convenient flat re-exports (mirrors gkd.* in the spec)
from .partition import Partition  # noqa: E402
from .vector import DistributedVector  # noqa: E402
from .matrix import DistributedMatrix  # noqa: E402
from .linop import PyLinOp  # noqa: E402

__all__ = [
    "Partition",
    "DistributedVector",
    "DistributedMatrix",
    "PyLinOp",
    "communicator",
    "partition",
    "vector",
    "matrix",
    "linop",
]
