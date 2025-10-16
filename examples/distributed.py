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

print(gkoComm.size)

partition = pGB.distributed.partition_from_global_size(executor, gkoComm.size, 10)
print(partition.size)

np_array = np.array([1, 2], dtype=np.float32)
local_vector = pGB.matrix.dense_float(np_array)
dist_vector = pGB.distributed.vector_float(executor, gkoComm, partition, local_vector)
