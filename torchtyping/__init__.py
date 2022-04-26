from .tensor_details import (
    DtypeDetail,
    is_float,
    is_named,
    LayoutDetail,
    ShapeDetail,
    TensorDetail,
)

from .tensor_type import JaxArray
from .typechecker import patch_typeguard, jit, typed_jit

__version__ = "0.1.4"
