# SPDX-License-Identifier: MIT
#
# SPDX-FileCopyrightText: 2024 pyGinkgo authors

import os
import sys
import pyGinkgoBindings as pgb
import pytest

sys.path.append("../build")


@pytest.mark.parametrize("solver", ["gmres"])
class TestSparseMatrix:
    ref = pgb.ReferenceExecutor()

    def test_can_create_solver(self, solver_name):
        reader = getattr(pgb.matrix, "read_coo")
        fn = os.path.dirname(os.path.realpath(__file__)) + "/sparse_example.mtx"
        # sparse = pgb.matrix.read_coo(fn, executor)
        sparse = reader(fn, self.ref)
        solver_ctr = getattr(pgb.solver, solver_name)
        # sparse = pgb.solver.gmres(executor, sparse_matrix, iter, reset, stop)
        solver = solver_ctr(self.ref, sparse, 100, 10, 1e-06)

        # to use the solver
        # rhs and sol ginkgo.dense
        # read rhs from disk
        # rhs = pyGinkgo.matrix.read_dense(fn, executor)
        # create x by copy construct and fill with zeros
        # solver.apply(rhs, sol)

        assert solver == solver

    def test_can_create_preconditioned_solver(self, solver_name):
        reader = getattr(pyGinkgo.matrix, "read_coo")
        fn = os.path.dirname(os.path.realpath(__file__)) + "/sparse_example.mtx"
        # sparse = pyGinkgo.matrix.read_coo(fn, executor)
        sparse = reader(fn, self.ref)
        solver_ctr = getattr(pyGinkgo.solver, solver_name)
        preconditioner = pyGinkgo.preconditioner.Ilu(self.ref, sparse)

        # sparse = pyGinkgo.solver.gmres(executor, sparse_matrix, iter, reset, stop)
        solver = solver_ctr(self.ref, sparse, preconditioner, 100, 10, 1e-06)

        # to use the solver
        # rhs and sol ginkgo.dense
        # read rhs from disk
        # rhs = pyGinkgo.matrix.read_dense(fn, executor)
        # create x by copy construct and fill with zeros
        # solver.apply(rhs, sol)

        assert solver == solver
