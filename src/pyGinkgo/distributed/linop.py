# SPDX-FileCopyrightText: 2024 - 2026 pyGinkgo authors
#
# SPDX-License-Identifier: MIT
"""Matrix-free PyLinOp facade."""

from .. import pyGinkgoBindings as _pgb

PyLinOp = _pgb.distributed.PyLinOp

__all__ = ["PyLinOp"]
