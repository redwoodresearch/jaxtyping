import pytest
from typeguard import typechecked
from typing import Tuple
import jax

from pathlib import Path
import sys
sys.path.append(Path(__file__).parent.resolve())
from torch_surrogate import TensorType
import torch_surrogate as torch


dim1 = dim2 = dim3 = channel = None


def test_non_tensor():
    class Tensor:
        shape = (2, 2)
        dtype = jax.numpy.float32
        layout = torch.strided

    args = (None, 4, 3.0, 3.2, "Tensor", Tensor, Tensor())

    @typechecked
    def accepts_tensor1(x: TensorType):
        pass

    @typechecked
    def accepts_tensor2(x: TensorType[2, 2]):
        pass

    @typechecked
    def accepts_tensor3(x: TensorType[...]):
        pass

    @typechecked
    def accepts_tensor4(x: TensorType[float]):
        pass

    @typechecked
    def accepts_tensor5(x: TensorType[..., float]):
        pass

    @typechecked
    def accepts_tensor6(x: TensorType[2, int]):
        pass

    with pytest.raises(TypeError):
        @typechecked
        def accepts_tensor7(x: TensorType[torch.strided]):
            pass

    with pytest.raises(TypeError):
        @typechecked
        def accepts_tensor8(x: TensorType[2, float, torch.sparse_coo]):
            pass

    for func in (
        accepts_tensor1,
        accepts_tensor2,
        accepts_tensor3,
        accepts_tensor4,
        accepts_tensor5,
        accepts_tensor6,
    ):
        for arg in args:
            with pytest.raises(TypeError):
                func(arg)

    @typechecked
    def accepts_tensors1(x: TensorType, y: TensorType):
        pass

    @typechecked
    def accepts_tensors2(x: TensorType[2, 2], y: TensorType):
        pass

    @typechecked
    def accepts_tensors3(x: TensorType[...], y: TensorType):
        pass

    @typechecked
    def accepts_tensors4(x: TensorType[float], y: TensorType):
        pass

    @typechecked
    def accepts_tensors5(x: TensorType[..., float], y: TensorType):
        pass

    @typechecked
    def accepts_tensors6(x: TensorType[2, int], y: TensorType):
        pass

    with pytest.raises(TypeError):
        @typechecked
        def accepts_tensors7(x: TensorType[torch.strided], y: TensorType):
            pass

    with pytest.raises(TypeError):
        @typechecked
        def accepts_tensors8(x: TensorType[torch.sparse_coo, float, 2], y: TensorType):
            pass

    for func in (
        accepts_tensors1,
        accepts_tensors2,
        accepts_tensors3,
        accepts_tensors4,
        accepts_tensors5,
        accepts_tensors6,
    ):
        for arg1 in args:
            for arg2 in args:
                with pytest.raises(TypeError):
                    func(arg1, arg2)


def test_nested_types():
    @typechecked
    def func(x: Tuple[TensorType[3, "channel", 4], TensorType["channel"]]):
        pass

    func((torch.rand(3, 1, 4), torch.rand(1)))
    func((torch.rand(3, 5, 4), torch.rand(5)))
    with pytest.raises(TypeError):
        func((torch.rand(3, 1, 4), torch.rand(2)))


def test_no_getitem():
    @typechecked
    def func(x: TensorType, y: TensorType):
        pass

    func(torch.rand(2), torch.rand(2))
    with pytest.raises(TypeError):
        func(torch.rand(2), None)
    with pytest.raises(TypeError):
        func(torch.rand(2), [3, 4])


def test_scalar_tensor():
    @typechecked
    def func(x: TensorType[()]):
        pass

    func(torch.rand(()))
    with pytest.raises(TypeError):
        func(torch.rand((1,)))
    with pytest.raises(TypeError):
        func(torch.rand((1, 2)))
    with pytest.raises(TypeError):
        func(torch.rand((5, 2, 2)))


def test_square():
    @typechecked
    def func(x: TensorType["dim1", "dim1"]):
        pass

    func(torch.rand(2, 2))
    func(torch.rand(5, 5))
    with pytest.raises(TypeError):
        func(torch.rand(3, 5))
    with pytest.raises(TypeError):
        func(torch.rand(5, 3))


def test_repeat():
    @typechecked
    def func(x: TensorType["dim1", "dim2", "dim2"], y: TensorType[-1, "dim2"]):
        pass

    func(torch.rand(5, 3, 3), torch.rand(9, 3))
    func(torch.rand(5, 5, 5), torch.rand(9, 5))
    func(torch.rand(4, 5, 5), torch.rand(2, 5))
    with pytest.raises(TypeError):
        func(torch.rand(4, 5, 4), torch.rand(3, 5))
    with pytest.raises(TypeError):
        func(torch.rand(4, 5, 5), torch.rand(3, 3))
    with pytest.raises(TypeError):
        func(torch.rand(4, 3, 5), torch.rand(3, 3))
    with pytest.raises(TypeError):
        func(torch.rand(4, 3, 3), torch.rand(0, 2))
