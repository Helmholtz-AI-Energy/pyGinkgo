# SPDX-FileCopyrightText: 2025 - 2026 pyGinkgo authors
#
# SPDX-License-Identifier: MIT

import enum
import numpy as np
from typing import Any
from typing import List, Union

from . import pyGinkgoBindings as pGB

class DatatypeEnum(str, enum.Enum):
    @staticmethod
    def _generate_next_value_(name, start, count, last_values) -> str:
        return name
    
    @classmethod
    def values(cls) -> List[str]:
        return [member.value for member in cls]


class npauto(enum.auto):
    def __init__(self, *args) -> None:
        self.__args = args
        self.__value = enum._auto_null
    
    @property
    def value(self):
        return self.__value
    
    @value.setter
    def value(self, value: Any) -> None:
        self.__value = ((value, *self.__args),)

class NpDatatypeEnum(DatatypeEnum):
    def __new__(cls, *args):
        if len(args) == 1:
            args = args[0]
        name, numpy_type = args

        member = str.__new__(cls, name)
        member._value_ = name
        member.numpy_type = numpy_type
        return member
    
    def __init__(self, *_):
        self.numpy_type: Any
    
    @classmethod
    def numpy_types(cls) -> List:
        return [member.numpy_type for member in cls]

    @classmethod
    def type_map(cls) -> dict:
        return {member.value: member.numpy_type for member in cls}
    
    def __str__(self) -> str:
        return f"{self.value}"


class ValueType(NpDatatypeEnum):
    half = npauto(np.float16) # type: ignore
    float = npauto(np.float32) # type: ignore
    double = npauto(np.float64) # type: ignore

class IndexType(NpDatatypeEnum):
    int32 = npauto(np.int32) # type: ignore
    # https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.longlong
    int64 = npauto(np.longlong) # type: ignore

dtype: List[NpDatatypeEnum] = [*ValueType, *IndexType]

class MatrixFormat(DatatypeEnum):
    dense = enum.auto()
    # TODO: it makes sense to make them the same casing
    Csr = "Csr"
    Coo = "Coo"
    

class ExecutorType(DatatypeEnum):
    cpu = enum.auto()
    omp = enum.auto()
    cuda = enum.auto()
    hip = enum.auto()
    dpcpp = enum.auto()

DeviceType = Union[ExecutorType, pGB.Executor, str]

# Reverse mappings: numpy dtype → corresponding Ginkgo type.
# CuPy dtypes are numpy-compatible, so these work for both.
NUMPY_TO_GKO_VALUE = {v.numpy_type: v for v in ValueType}
NUMPY_TO_GKO_INDEX = {v.numpy_type: v for v in IndexType}
