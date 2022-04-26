from .tensor_details import (
    DtypeDetail,
    is_float,
    LayoutDetail,
    ShapeDetail,
    TensorDetail,
)

from .tensor_type import JaxArray, jit
from .typechecker import patch_typeguard

__version__ = "0.1.4"
