# SPDX-FileCopyrightText: 2024 - 2025 pyGinkgo authors
#
# SPDX-License-Identifier: MIT

from mpi4py import MPI

import numpy as np
import pyGinkgo.pyGinkgoBindings as pGB

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

gkoComm = pGB.distributed.Communicator(False)
executor = pGB.ReferenceExecutor()

print("gkoComm.size", gkoComm.size)

partition = pGB.distributed.partition_from_global_size(executor, gkoComm.size, 10)
print(partition.size)


np_array = np.array([2, 2], dtype=np.float32)
local_b = pGB.matrix.dense_float(np_array)
dist_b = pGB.distributed.vector_float(executor, gkoComm, partition, local_b)

np_array = np.array([0, 0], dtype=np.float32)
local_x = pGB.matrix.dense_float(np_array)
dist_x = pGB.distributed.vector_float(executor, gkoComm, partition, local_x)

# rank 0 matrix builder
# [2 1 |    ...  ]
# [1 2 | 1  ...  ]
# [--------------]
# [  1 | 2 1 |   ]
# [    | 1 2 | 1 ]
# [--------------]
# [    | ... |   ]

# in_mtx = pGB.matrix.read_Coo_float_int32("examples/m1.mtx", executor) if rank==0 else pGB.matrix.Coo_float_int32(executor)
in_mtx = (
    pGB.matrix.read_Coo_float_int32("examples/m1.mtx", executor)
    if rank == 0
    else pGB.matrix.Coo_float_int32(executor)
)
print(dir(pGB.distributed))
dist_A = pGB.distributed.dist_matrix_object_float_int(
    executor, gkoComm, partition, in_mtx
)

dist_A.apply(dist_b, dist_x)
