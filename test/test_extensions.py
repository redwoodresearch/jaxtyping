import pytest
import jax.numpy as jnp
from jaxtyping import TensorDetail, JaxArray
from typeguard import typechecked

from pathlib import Path
import sys
sys.path.append(Path(__file__).parent.resolve())
from torch_surrogate import rand, Tensor, TensorType

good = foo = None

# Write the extension


class KindDetail(TensorDetail):
    def __init__(self, kind):
        super().__init__()
        self.kind = kind

    def check(self, array: JaxArray) -> bool:
        return array.dtype.kind == self.kind

    # reprs used in error messages when the check is failed

    def __repr__(self) -> str:
        return f"KindDetail({self.kind})"

    @classmethod
    def tensor_repr(cls, array: JaxArray) -> str:
        # Should return a representation of the array with respect
        # to what this detail is checking
        if array.dtype.kind == "u":
            return f"KindDetail({array.dtype.kind})"
        else:
            return ""


# Test the extension


@typechecked
def foo_checker(tensor: TensorType[3, KindDetail("u")]):
    pass


def valid_foo():
    x = rand(3).astype(jnp.uint32)
    foo_checker(x)


def invalid_foo_one():
    x = rand(3).astype(jnp.float16)
    foo_checker(x)


def invalid_foo_two():
    x = rand(2).astype(jnp.uint8)
    foo_checker(x)


def test_extensions():
    valid_foo()
    with pytest.raises(TypeError):
        invalid_foo_one()
    with pytest.raises(TypeError):
        invalid_foo_two()
