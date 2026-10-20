# SPDX-FileCopyrightText: 2024 - 2026 pyGinkgo authors
#
# SPDX-License-Identifier: MIT

import os
import pytest
import numpy as np

import pyGinkgo as pg
import pyGinkgo.pyGinkgoBindings as pGB

from test_utils import verify_dense_vec

d_precision_map = {
    "half": 1e-3,
    "float": 1e-6,
    "double": 0,
}


@pytest.mark.parametrize("data_type", list(pg.gko_types.ValueType))
class TestDense:
    values = [1.0, 2.0, -1.0, 3.0, 4.0, -1.0, 5.0, 6.0, -1.0]
    values2d = [[1.0, 2.0, -1.0], [3.0, 4.0, -1.0], [5.0, 6.0, -1.0]]
    values2d2 = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
    ref = pGB.ReferenceExecutor()

    def test_can_create_dense_linop(self, data_type: pg.gko_types.ValueType):
        dense_cls = getattr(pGB.matrix, "dense_" + data_type)
        dense = dense_cls(self.ref)
        assert dense == dense

    def test_can_create_dense_linop_with_dim(self, data_type: pg.gko_types.ValueType):
        dense_cls = getattr(pGB.matrix, "dense_" + data_type)
        dense = dense_cls(self.ref, (len(self.values), 1))
        dense.fill(42.0)
        assert dense == dense
        assert dense.at(0, 0) == 42.0

    def test_can_create_dense_linop_with_dim_stride(
        self, data_type: pg.gko_types.ValueType
    ):
        dense_cls = getattr(pGB.matrix, "dense_" + data_type)
        dense = dense_cls(self.ref, (len(self.values), 1), 1)
        assert dense == dense

    def test_can_create_dense_from_1D_np_array_with_default_exec(
        self, data_type: pg.gko_types.ValueType
    ):
        dense_cls = getattr(pGB.matrix, "dense_" + data_type)
        dense = dense_cls(np.array(self.values, dtype=data_type.numpy_type))
        verify_dense_vec(dense, self.values)

    def test_can_copy_construct(self, data_type: pg.gko_types.ValueType):
        dense_cls = getattr(pGB.matrix, "dense_" + data_type)
        dense_a = dense_cls(np.array(self.values, dtype=data_type.numpy_type))
        dense_b = dense_cls(self.ref, dense_a)
        verify_dense_vec(dense_b, self.values)

    def test_can_create_dense_from_1D_np_array(self, data_type: pg.gko_types.ValueType):
        dense_cls = getattr(pGB.matrix, "dense_" + data_type)
        dense = dense_cls(self.ref, np.array(self.values, dtype=data_type.numpy_type))
        verify_dense_vec(dense, self.values)

    def test_can_create_dense_from_1D_np_array_with_dim_stride(
        self, data_type: pg.gko_types.ValueType
    ):
        dense_cls = getattr(pGB.matrix, "dense_" + data_type)
        dense = dense_cls(
            self.ref,
            (len(self.values), 1),
            np.array(self.values, dtype=data_type.numpy_type),
            1,
        )
        verify_dense_vec(dense, self.values)

    def test_can_read_mtx_file(self, data_type: pg.gko_types.ValueType):
        # Matrix market format stores column major order
        fn = os.path.dirname(os.path.realpath(__file__)) + "/dense_example.mtx"
        read_func = getattr(pGB.matrix, "read_dense_" + data_type)
        dense = read_func(fn, self.ref)

        assert dense.get_num_stored_elements() == 12
        assert dense.at(0, 2) == pytest.approx(0.2785, abs=d_precision_map[data_type])
        assert dense.at(2) == pytest.approx(0.2785, abs=d_precision_map[data_type])
        assert dense.at(2, 2) == pytest.approx(0.9575, abs=d_precision_map[data_type])

    def test_can_create_dense_from_2D_np_array_with_default_exec(
        self, data_type: pg.gko_types.ValueType
    ):
        dense_cls = getattr(pGB.matrix, "dense_" + data_type)
        dense = dense_cls(np.array(self.values2d, dtype=data_type.numpy_type))
        assert dense.at(0, 1) == 2.0
        assert dense.at(0, 2) == -1.0
        assert dense.at(1, 0) == 3.0
        assert dense.at(1, 1) == 4.0
        assert dense.at(1, 2) == -1.0

    def test_can_create_dense_from_2D2_np_array_with_default_exec(
        self, data_type: pg.gko_types.ValueType
    ):
        dense_cls = getattr(pGB.matrix, "dense_" + data_type)
        np_array = np.array(self.values2d2, dtype=data_type.numpy_type)
        dense = dense_cls(np_array)
        assert dense.at(0, 0) == np_array[0][0]
        assert dense.at(0, 1) == np_array[0][1]
        assert dense.at(1, 0) == np_array[1][0]
        assert dense.at(1, 1) == np_array[1][1]
        assert dense.at(2, 0) == np_array[2][0]
        assert dense.at(2, 1) == np_array[2][1]

        # test if conversion is invariant
        np_array_in = np.array(self.values2d2, dtype=data_type.numpy_type)
        np_array_out = np.array(dense, dtype=data_type.numpy_type)

        assert (np_array_out == np_array_in).all()

    def test_can_create_dense_from_1D_np_array_with_stride(
        self, data_type: pg.gko_types.ValueType
    ):
        dense_cls = getattr(pGB.matrix, "dense_" + data_type)
        dense = dense_cls(
            self.ref, (3, 3), np.array([self.values], dtype=data_type.numpy_type), 3
        )

        assert dense.get_num_stored_elements() == len(self.values)
        assert dense.at(2) == -1.0
        assert dense.at(0, 2) == dense.at(2)
        assert dense.at(2, 1) == 6

    def test_dense_support_basic_functionality(self, data_type: pg.gko_types.ValueType):
        dense_cls = getattr(pGB.matrix, "dense_" + data_type)
        dense = dense_cls(
            self.ref, (3, 3), np.array([self.values], dtype=data_type.numpy_type), 3
        )

        dense.scale(5)
        assert dense.at(2) == -5.0
        assert dense.at(1, 2) == dense.at(5)

        dense.inv_scale(5)
        assert dense.at(2) == -1.0
        assert dense.at(1, 2) == dense.at(5)

    def test_can_apply_to_transpose(self, data_type: pg.gko_types.ValueType):
        dense_cls = getattr(pGB.matrix, "dense_" + data_type)
        aT = np.array([self.values], dtype=data_type.numpy_type)
        a = aT.T
        dense_a = dense_cls(aT)
        dense_b = dense_cls(a)
        result = dense_cls(self.ref, (1, 1))
        dense_a.apply(dense_b, result)
        assert result.at(0) == aT.dot(a)[0, 0]

    def test_can_transpose(self, data_type: pg.gko_types.ValueType):
        dense_cls = getattr(pGB.matrix, "dense_" + data_type)
        dense_a = dense_cls(
            self.ref, (3, 3), np.array([self.values], dtype=data_type.numpy_type), 3
        )
        dense_aT = dense_a.T()
        dense_a.at(0, 1) == dense_aT.at(1, 0)
        dense_a.at(0, 2) == dense_aT.at(2, 0)
        dense_a.at(1, 2) == dense_aT.at(2, 1)
        dense_a.at(0, 0) == dense_aT.at(0, 0)
        dense_a.at(1, 1) == dense_aT.at(1, 1)
        dense_a.at(2, 2) == dense_aT.at(2, 2)

    def test_add_scaled(self, data_type: pg.gko_types.ValueType):
        dense_cls = getattr(pGB.matrix, "dense_" + data_type)
        a = np.array(self.values, dtype=data_type.numpy_type)
        dense_a = dense_cls(a)
        alpha = 2.0
        dense_alpha = dense_cls(np.array([alpha], dtype=data_type.numpy_type))
        dense_a.add_scaled(dense_alpha, dense_a)
        verify_dense_vec(dense_a, (1 + alpha) * a)

    def test_sub_scaled(self, data_type: pg.gko_types.ValueType):
        dense_cls = getattr(pGB.matrix, "dense_" + data_type)
        a = np.array(self.values, dtype=data_type.numpy_type)
        dense_a = dense_cls(a)
        alpha = 0.5
        dense_alpha = dense_cls(np.array([alpha], dtype=data_type.numpy_type))
        dense_a.sub_scaled(dense_alpha, dense_a)
        verify_dense_vec(dense_a, (1 - alpha) * a)

    def test_dense_can_return_size(self, data_type: pg.gko_types.ValueType):
        dense_cls = getattr(pGB.matrix, "dense_" + data_type)
        dense = dense_cls(
            self.ref, (3, 3), np.array(self.values, dtype=data_type.numpy_type), 3
        )
        with pytest.deprecated_call():
            assert dense.get_size()[0] == 3
            assert dense.get_size()[1] == 3

    def test_dense_size_property(self, data_type: pg.gko_types.ValueType):
        dense_cls = getattr(pGB.matrix, "dense_" + data_type)
        dense = dense_cls(
            self.ref, (3, 3), np.array(self.values, dtype=data_type.numpy_type), 3
        )
        with pytest.deprecated_call():
            assert dense.size[0] == 3
            assert dense.size[1] == 3
            with pytest.raises(AttributeError):
                dense.size = pGB.dim2(4, 4)

    def test_dense_shape_property(self, data_type: pg.gko_types.ValueType):
        dense_cls = getattr(pGB.matrix, "dense_" + data_type)
        dense = dense_cls(
            self.ref, (3, 3), np.array(self.values, dtype=data_type.numpy_type), 3
        )
        assert dense.shape == (3, 3)
        with pytest.raises(AttributeError):
            dense.shape = (4, 4)

    def test_dense_column_scale(self, data_type: pg.gko_types.ValueType):
        dense_cls = getattr(
            pGB.matrix,
            "dense_" + data_type,
        )
        np_type = data_type.numpy_type

        x_np = np.array(
            [
                [1.0, 2.0],
                [3.0, 4.0],
                [5.0, 6.0],
            ],
            dtype=np_type,
        )

        x = dense_cls(self.ref, x_np)

        alpha = dense_cls(
            self.ref,
            np.array(
                [[2.0, 3.0]],
                dtype=np_type,
            ),
        )

        x.scale(alpha)

        np.testing.assert_allclose(
            np.asarray(x.copy_to_host()),
            x_np
            * np.array(
                [2.0, 3.0],
                dtype=np_type,
            ),
            rtol=d_precision_map[data_type],
            atol=d_precision_map[data_type],
        )

    def test_dense_lobpcg_operations(self, data_type: pg.gko_types.ValueType):
        executor = pGB.ReferenceExecutor()

        dense_cls = getattr(pGB.matrix, "dense_" + data_type)
        np_type = data_type.numpy_type

        values = np.array(
            [
                [1.0, 2.0],
                [3.0, 4.0],
                [5.0, 6.0],
            ],
            dtype=np_type,
        )

        x = dense_cls(executor, values)

        # clone()
        cloned = x.clone()
        assert cloned.shape == x.shape

        np.testing.assert_allclose(
            np.asarray(cloned.copy_to_host()),
            values,
            rtol=d_precision_map[data_type],
            atol=d_precision_map[data_type],
        )

        # create_with_config_of()
        work = x.create_with_config_of()
        assert work.shape == x.shape

        # copy_from()
        work.copy_from(x)

        np.testing.assert_allclose(
            np.asarray(work.copy_to_host()),
            values,
            rtol=d_precision_map[data_type],
            atol=d_precision_map[data_type],
        )

        # create_with_type_of() + transpose_into()
        transpose = x.create_with_type_of((2, 3))
        x.transpose_into(transpose)

        np.testing.assert_allclose(
            np.asarray(transpose.copy_to_host()),
            values.T,
            rtol=d_precision_map[data_type],
            atol=d_precision_map[data_type],
        )

        # Column-wise scale.
        scale = dense_cls(
            executor,
            np.array([[2.0, 3.0]], dtype=np_type),
        )

        work.copy_from(x)
        work.scale(scale)

        np.testing.assert_allclose(
            np.asarray(work.copy_to_host()),
            values * np.array([2.0, 3.0], dtype=np_type),
            rtol=d_precision_map[data_type],
            atol=d_precision_map[data_type],
        )

        # compute_norm2()
        norms = x.create_with_type_of((1, 2))
        x.compute_norm2(norms)

        np.testing.assert_allclose(
            np.asarray(norms.copy_to_host()).reshape(-1),
            np.linalg.norm(values, axis=0),
            rtol=d_precision_map[data_type],
            atol=d_precision_map[data_type],
        )

    def test_dense_advanced_apply_with_python_scalars(
        self, data_type: pg.gko_types.ValueType
    ):
        dense_cls = getattr(pGB.matrix, "dense_" + data_type)
        np_type = data_type.numpy_type

        A = dense_cls(
            self.ref,
            np.eye(2, dtype=np_type),
        )

        b = dense_cls(
            self.ref,
            np.array(
                [
                    [1.0],
                    [2.0],
                ],
                dtype=np_type,
            ),
        )

        x = dense_cls(
            self.ref,
            np.array(
                [
                    [10.0],
                    [20.0],
                ],
                dtype=np_type,
            ),
        )

        A.apply(2.0, b, 3.0, x)

        np.testing.assert_allclose(
            np.array(x.copy_to_host()).reshape(-1),
            np.array([32.0, 64.0], dtype=np_type),
            rtol=d_precision_map[data_type],
            atol=d_precision_map[data_type],
        )
