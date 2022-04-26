import pytest
import torch
from jaxtyping import is_float, is_named
import typeguard

from pathlib import Path
import sys
sys.path.append(Path(__file__).parent.resolve())
from torch_surrogate import TensorType, skip_named_test
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

@skip_named_test
def test_named_tensor():
    @typeguard.typechecked
    def _named_a_dim_checker(x: TensorType["dim1", is_named]):
        pass

    @typeguard.typechecked
    def _named_ab_dim_checker(x: TensorType["dim1", "dim2", is_named]):
        pass

    @typeguard.typechecked
    def _named_abc_dim_checker(x: TensorType["dim1", "dim2", "dim3", is_named]):
        pass

    @typeguard.typechecked
    def _named_cb_dim_checker(x: TensorType["dim3", "dim2", is_named]):
        pass

    @typeguard.typechecked
    def _named_am1_dim_checker(x: TensorType["dim1", -1, is_named]):
        pass

    @typeguard.typechecked
    def _named_m1b_dim_checker(x: TensorType[-1, "dim2", is_named]):
        pass

    @typeguard.typechecked
    def _named_abm1_dim_checker(x: TensorType["dim1", "dim2", -1, is_named]):
        pass

    @typeguard.typechecked
    def _named_m1bm1_dim_checker(x: TensorType[-1, "dim2", -1, is_named]):
        pass

    x = torch.rand(3, 4)
    named_x = torch.rand(3, 4, names=("dim1", "dim2"))

    with pytest.raises(TypeError):
        _named_ab_dim_checker(x)
    with pytest.raises(TypeError):
        _named_cb_dim_checker(x)
    with pytest.raises(TypeError):
        _named_am1_dim_checker(x)
    with pytest.raises(TypeError):
        _named_m1b_dim_checker(x)
    with pytest.raises(TypeError):
        _named_a_dim_checker(x)
    with pytest.raises(TypeError):
        _named_abc_dim_checker(x)
    with pytest.raises(TypeError):
        _named_abm1_dim_checker(x)
    with pytest.raises(TypeError):
        _named_m1bm1_dim_checker(x)

    _named_ab_dim_checker(named_x)
    _named_am1_dim_checker(named_x)
    _named_m1b_dim_checker(named_x)
    with pytest.raises(TypeError):
        _named_a_dim_checker(named_x)
    with pytest.raises(TypeError):
        _named_abc_dim_checker(named_x)
    with pytest.raises(TypeError):
        _named_cb_dim_checker(named_x)
    with pytest.raises(TypeError):
        _named_abm1_dim_checker(named_x)
    with pytest.raises(TypeError):
        _named_m1bm1_dim_checker(named_x)


@skip_named_test
def test_named_float_tensor():
    @typeguard.typechecked
    def func(x: TensorType["dim1", "dim2":3, is_float, is_named]):
        pass

    x = torch.rand(2, 3, names=("dim1", "dim2"))
    y = torch.rand(2, 2, names=("dim1", "dim2"))
    z = torch.rand(2, 2, names=("dim1", "dim3"))
    w = torch.rand(2, 3)
    w1 = torch.rand(2, 2, names=("dim1", None))
    w2 = torch.rand(2, 3, names=("dim1", "dim2")).int()

    func(x)
    with pytest.raises(TypeError):
        func(y)
    with pytest.raises(TypeError):
        func(z)
    with pytest.raises(TypeError):
        func(w)
    with pytest.raises(TypeError):
        func(w1)
    with pytest.raises(TypeError):
        func(w2)


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


@skip_named_test
def test_named_ellipsis():
    @typeguard.typechecked
    def func(x: TensorType["dim1":..., "dim2", is_named]):
        pass

    func(torch.rand(3, 4, names=(None, "dim2")))
    func(torch.rand(3, 4, names=("another_dim", "dim2")))
    with pytest.raises(TypeError):
        func(torch.rand(3, 4))
    with pytest.raises(TypeError):
        func(torch.rand(3, 4, names=(None, None)))
    with pytest.raises(TypeError):
        func(torch.rand(3, 4, names=("dim2", None)))
