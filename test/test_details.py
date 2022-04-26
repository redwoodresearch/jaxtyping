import pytest
import torch
from jaxtyping import is_float
import typeguard

from pathlib import Path
import sys
sys.path.append(Path(__file__).parent.resolve())
from torch_surrogate import TensorType
import torch_surrogate as torch

dim1 = dim2 = dim3 = None


def test_float_tensor():
    @typeguard.typechecked
    def func1(x: TensorType[is_float]):
        pass

    @typeguard.typechecked
    def func2(x: TensorType[2, 2, is_float]):
        pass

    @typeguard.typechecked
    def func3(x: TensorType[float, is_float]):
        pass

    @typeguard.typechecked
    def func4(x: TensorType[bool, is_float]):
        pass

    @typeguard.typechecked
    def func5(x: TensorType["dim1":2, 2, float, is_float]):
        pass

    with pytest.raises(TypeError):
        @typeguard.typechecked
        def func6(x: TensorType[2, "dim2":2, torch.sparse_coo, is_float]):
            pass

    x = torch.rand(2, 2)
    y = torch.rand(1)
    z = torch.tensor([[0, 1], [2, 3]])

    func1(x)
    func1(y)
    with pytest.raises(TypeError):
        func1(z)

    func2(x)
    with pytest.raises(TypeError):
        func2(y)
    with pytest.raises(TypeError):
        func2(z)

    func3(x)
    func3(y)
    with pytest.raises(TypeError):
        func3(z)

    with pytest.raises(TypeError):
        func4(x)
    with pytest.raises(TypeError):
        func4(y)
    with pytest.raises(TypeError):
        func4(z)

    func5(x)
    with pytest.raises(TypeError):
        func5(y)
    with pytest.raises(TypeError):
        func5(z)


def test_none_names():
    @typeguard.typechecked
    def func_unnamed1(x: TensorType[None:4]):
        pass

    @typeguard.typechecked
    def func_unnamed2(x: TensorType[None:4, "dim1"]):
        pass

    @typeguard.typechecked
    def func_unnamed3(x: TensorType[None:4, "dim1"], y: TensorType["dim1", None]):
        pass

    func_unnamed1(torch.rand(4))
    with pytest.raises(TypeError):
        func_unnamed1(torch.rand(5))
    with pytest.raises(TypeError):
        func_unnamed1(torch.rand(2, 3))

    func_unnamed2(torch.rand(4, 5))

    func_unnamed3(torch.rand(4, 5), torch.rand(5, 3))
    with pytest.raises(TypeError):
        func_unnamed3(torch.rand(4, 5), torch.rand(3, 3))
