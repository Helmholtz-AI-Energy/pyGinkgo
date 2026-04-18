# SPDX-FileCopyrightText: 2024 - 2026 pyGinkgo authors
#
# SPDX-License-Identifier: MIT

from .pyGinkgoBindings import \
    base, factorization, logger, matrix

from .core import *
from .device import *
from .rayleigh_ritz import *

from . import gko_types
from . import solver
from . import preconditioner

# Optionally expose the distributed (MPI) subpackage. It self-checks for
# MPI build flags and mpi4py at import; if anything is missing we keep
# the rest of pyGinkgo usable.
try:
    from . import distributed  # noqa: F401
except ImportError:
    pass
