# SPDX-License-Identifier: MIT
#
# SPDX-FileCopyrightText: 2024 pyGinkgo authors

import sys
import numpy as np

sys.path.append("../../")
import pyGinkgoBindings as pgb


class TestFactorizationBinding:
    ref = pgb.ReferenceExecutor()
    values = [[2.0, 1.0, 1.0], [1.0, 2.0, 1.0], [1.0, 1.0, 2.0]]

    def test_factorization(self):
        dense = pgb.matrix.dense(self.ref, np.array(self.values))
        factorization = pgb.factorization.Factorization(self.ref, dense)
        lower = factorization.get_lower_factor()
        lower_np = np.array(lower)
        assert lower_np == np.array(self.values)
