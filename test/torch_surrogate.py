import jax.numpy as jnp
import jaxtyping
import warnings
import pytest
import functools

TensorType = jaxtyping.JaxArray
Tensor = jnp.ndarray

def ones(*shape, names=None):
    if len(shape) == 1:
        return jnp.ones(shape[0])
    return jnp.ones(shape)

def rand(*shape, names=None):
    if len(shape) == 1:
        return jnp.zeros(shape[0])
    return jnp.zeros(shape)

def tensor(data, names=None):
    return jnp.array(data)

sparse_coo = NotImplemented
strided = NotImplemented


def skip_named_test(test):
    warnings.warn(f"Test {test} will be skipped due to is_named")

    @functools.wraps(test)
    def new_test(*args, **kwargs):
        with pytest.raises(TypeError) as err:
            out = test(*args, **kwargs)
            assert str(err) == "There are no named JaxArrays"
            return out
    return new_test

def skip_layout_test(test):
    warnings.warn(f"Test {test} will be skipped due to using `sparse_coo` or `strided`")

    @functools.wraps(test)
    def new_test(*args, **kwargs):
        with pytest.raises(TypeError):
            return test(*args, **kwargs)
    return new_test
