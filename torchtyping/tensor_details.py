from __future__ import annotations

import abc
import collections
import jax.numpy
import jax.core

from typing import Optional, Union

JaxArray = Union[jax.numpy.ndarray, jax.core.UnshapedArray]

ellipsis = type(...)


class TensorDetail(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __repr__(self) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    def check(self, tensor: JaxArray) -> bool:
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def tensor_repr(cls, tensor: JaxArray) -> str:
        raise NotImplementedError


_no_name = object()


# inheriting from typing.NamedTuple crashes typeguard
class _Dim(collections.namedtuple("_Dim", ["name", "size"])):
    # None corresponds to a name not being set. no_name corresponds to us not caring
    # whether a name is set.
    name: Union[None, str, type(_no_name)]
    # technically supposed to use an enum to annotate singletons but that's overkill.

    size: Union[ellipsis, int]

    def __repr__(self) -> str:
        if self.name is _no_name:
            if self.size is ...:
                return "..."
            else:
                return repr(self.size)
        else:
            if self.size is ...:
                return f"{self.name}: ..."
            elif self.size == -1:
                return repr(self.name)
            else:
                return f"{self.name}: {self.size}"


class ShapeDetail(TensorDetail):
    def __init__(self, *, dims: list[_Dim], check_names: bool, **kwargs) -> None:
        super().__init__(**kwargs)
        self.dims = dims
        if check_names:
            raise TypeError("There are no named JaxArrays.")
        self.check_names = check_names

    def __repr__(self) -> str:
        if len(self.dims) == 0:
            out = "()"
        elif len(self.dims) == 1:
            out = repr(self.dims[0])
        else:
            out = repr(tuple(self.dims))[1:-1]
        if self.check_names:
            out += ", is_named"
        return out

    def check(self, tensor: JaxArray) -> bool:
        self_names = [self_dim.name for self_dim in self.dims]
        self_shape = [self_dim.size for self_dim in self.dims]

        if ... in self_shape:
            if sum(1 for size in self_shape if size is not ...) > len(tensor.shape):
                return False
        else:
            if len(self_shape) != len(tensor.shape):
                return False

        for self_name, self_size, tensor_size in zip(
            reversed(self_names),
            reversed(self_shape),
            reversed(tensor.shape),
        ):
            if self_size is ...:
                break

            if (
                self.check_names
                and self_name is not _no_name
                and self_name != tensor_name
            ):
                return False
            if not isinstance(self_size, str) and self_size not in (-1, tensor_size):
                return False

        return True

    @classmethod
    def tensor_repr(cls, tensor: JaxArray) -> str:
        dims = []
        # check_names = any(name is not None for name in tensor.names)
        check_names = False
        for size in tensor.shape:
            if not check_names:
                name = _no_name
            dims.append(_Dim(name=name, size=size))
        return repr(cls(dims=dims, check_names=check_names))

    def update(
        self,
        *,
        dims: Optional[list[_Dim]] = None,
        check_names: Optional[bool] = None,
        **kwargs,
    ) -> ShapeDetail:
        dims = self.dims if dims is None else dims
        check_names = self.check_names if check_names is None else check_names
        return type(self)(dims=dims, check_names=check_names, **kwargs)


class DtypeDetail(TensorDetail):
    def __init__(self, *, dtype, **kwargs) -> None:
        super().__init__(**kwargs)
        assert isinstance(dtype, (jax.numpy.dtype, type(jax.numpy.int32)))
        self.dtype = dtype

    def __repr__(self) -> str:
        return repr(self.dtype)

    def check(self, tensor: JaxArray) -> bool:
        return self.dtype == tensor.dtype

    @classmethod
    def tensor_repr(cls, tensor: JaxArray) -> str:
        return repr(cls(dtype=tensor.dtype))


class _FloatDetail(TensorDetail):
    def __repr__(self) -> str:
        return "is_float"

    def check(self, tensor: JaxArray) -> bool:
        return tensor.dtype.kind == "f"

    @classmethod
    def tensor_repr(cls, tensor: JaxArray) -> str:
        return "is_float" if (tensor.dtype.kind == "f") else ""


# is_named is special-cased and consumed by JaxArray.
# It's a bit of an odd exception.
# It's only a TensorDetail for consistency, as the other
# extra flags that get passed are TensorDetails.
class _NamedTensorDetail(TensorDetail):
    def __repr__(self) -> str:
        raise RuntimeError

    def check(self, tensor: JaxArray) -> bool:
        raise RuntimeError

    @classmethod
    def tensor_repr(cls, tensor: JaxArray) -> str:
        raise RuntimeError


is_float = _FloatDetail()  # singleton flag
is_named = _NamedTensorDetail()  # singleton flag
