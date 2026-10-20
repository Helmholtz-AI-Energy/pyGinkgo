# SPDX-FileCopyrightText: 2024 - 2026 pyGinkgo authors
#
# SPDX-License-Identifier: MIT

import os as _os
import sys as _sys

# The Ginkgo shared libraries ship next to the extension module. On Windows a
# DLL's own imports are resolved by name against the default search path, which
# does not include this package directory, so loading ginkgo.dll cold fails even
# though the ginkgo_* libraries it imports sit right beside it. Note that
# os.add_dll_directory does not help: CPython loads extension modules with
# LOAD_WITH_ALTERED_SEARCH_PATH, which does not consult those directories.
#
# Pre-loading each bundled library by absolute path does work -- once a library
# is in the process, later imports of it resolve by name. The link order is not
# known here, so keep retrying until a pass loads nothing new.
if _sys.platform == "win32":
    import ctypes as _ctypes
    import glob as _glob

    _package_dir = _os.path.dirname(_os.path.abspath(__file__))
    if hasattr(_os, "add_dll_directory"):
        _os.add_dll_directory(_package_dir)

    _pending = _glob.glob(_os.path.join(_package_dir, "*.dll"))
    while _pending:
        _unresolved = []
        for _library in _pending:
            try:
                _ctypes.WinDLL(_library)
            except OSError:
                _unresolved.append(_library)
        if len(_unresolved) == len(_pending):
            # No progress this pass; let the import below surface the error.
            break
        _pending = _unresolved

from .pyGinkgoBindings import \
    base, factorization, logger, matrix

from .core import *
from .device import *
from .rayleigh_ritz import *

from . import gko_types
from . import solver
from . import preconditioner
from . import distributed
