# SPDX-FileCopyrightText: 2024 - 2025 pyGinkgo authors
#
# SPDX-License-Identifier: MIT

import ctypes
import glob
import os
import sys


def _preload_libs():
    """Preload Ginkgo shared libraries so they can be found without
    setting LD_LIBRARY_PATH manually.

    The Ginkgo shared libraries (libginkgo, libginkgo_device, …) are
    installed alongside the Python extension module.  On Linux the
    dynamic linker only searches a fixed set of paths unless
    LD_LIBRARY_PATH is set.  By explicitly loading every Ginkgo library
    with ``ctypes.CDLL`` (using ``RTLD_GLOBAL`` so their symbols are
    visible to later loads) we make sure they are available when the
    extension module is imported – no environment-variable setup needed.
    """
    if sys.platform.startswith("linux"):
        ext = ".so"
    elif sys.platform == "darwin":
        ext = ".dylib"
    else:
        return

    pkg_dir = os.path.dirname(os.path.abspath(__file__))

    # Load in dependency order: device first, then backends, then the
    # main library (which depends on all the others).
    _lib_load_order = [
        "libginkgo_device",
        "libginkgo_reference",
        "libginkgo_omp",
        "libginkgo_cuda",
        "libginkgo_hip",
        "libginkgo_dpcpp",
        "libginkgo",
    ]

    for lib_name in _lib_load_order:
        pattern = os.path.join(pkg_dir, lib_name + ext + "*")
        for lib_path in sorted(glob.glob(pattern)):
            try:
                ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                # Backend not available (e.g. no CUDA) – that is fine.
                pass


_preload_libs()

from .pyGinkgoBindings import \
    base, factorization, logger, matrix

from .core import *
from .device import *
from .rayleigh_ritz import *

from . import gko_types
from . import solver
from . import preconditioner
from . import cupy_interop

