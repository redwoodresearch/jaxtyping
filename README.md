<h1 align='center'>jaxtyping</h1>
<h2 align='center'>Type annotations for an array's shape, dtype, ...</h2>

Like [Torchtyping](https://github.com/patrick-kidger/torchtyping), but for Jax.

Turn this:
```python
def batch_outer_product(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    # x has shape (batch, x_channels)
    # y has shape (batch, y_channels)
    # return has shape (batch, x_channels, y_channels)

    return x[:, None] * y[None, :]
```
into this:
```python
def batch_outer_product(x:   JaxArray["batch", "x_channels"],
                        y:   JaxArray["batch", "y_channels"]
                        ) -> JaxArray["batch", "x_channels", "y_channels"]:

    return x[:, None] * y[None, :]
```
**with programmatic checking that the shape (dtype, ...) specification is met.**

Bye-bye bugs! Say hello to enforced, clear documentation of your code.

If (like me) you find yourself littering your code with comments like `# x has shape (batch, hidden_state)` or statements like `assert x.shape == y.shape` , just to keep track of what shape everything is, **then this is for you.**

---

## Installation

```bash
pip install jaxtyping
```

Requires Python 3.7+ and Jax.

## Usage

`jaxtyping` allows for type annotating:

- **shape**: size, number of dimensions;
- **dtype** (float, integer, etc.);
- **names** of dimensions. There is no support for [named
  tensors](https://jax.readthedocs.io/en/latest/_autosummary/jax.experimental.maps.xmap.html)
  yet, but `jaxtyping` can check that dimensions with the same name match.
- **arbitrary number of batch dimensions** with `...`;
- **...plus anything else you like**, as `jaxtyping` is highly extensible.

If [`typeguard`](https://github.com/agronholm/typeguard) is (optionally) installed then **at runtime the types can be checked** to ensure that the tensors really are of the advertised shape, dtype, etc. 

```python
# EXAMPLE

from jax import jit
from jax.numpy import zeros
from jaxtyping import TensorType, patch_typeguard
from typeguard import typechecked

patch_typeguard()  # use before @typechecked

@jit  # Type check only when compiling
@typechecked
def func(x: JaxArray["batch"],
         y: JaxArray["batch"]) -> JaxArray["batch"]:
    return x + y

func(zeros(3), zeros(3))  # works
func(zeros(3), zeros(1))
# TypeError: Dimension 'batch' of inconsistent size. Got both 1 and 3.
```

`typeguard` also has an import hook that can be used to automatically test an entire module, without needing to manually add `@typeguard.typechecked` decorators.

If you're not using `typeguard` then `jaxtyping.patch_typeguard()` can be omitted altogether, and `jaxtyping` just used for documentation purposes. If you're not already using `typeguard` for your regular Python programming, then strongly consider using it. It's a great way to squash bugs. Both `typeguard` and `jaxtyping` also integrate with `pytest`, so if you're concerned about any performance penalty then they can be enabled during tests only.

## API

```python
jaxtyping.JaxArray[shape1, shape2, ...shapeN, dtype, details]
```

The core of the library.

Each of `shape`, `dtype`, `details` are optional.

- The `shape` argument can be any of:
  - An `int`: the dimension must be of exactly this size. If it is `-1` then any size is allowed.
  - A `str`: the size of the dimension passed at runtime will be bound to this name, and all tensors checked that the sizes are consistent.
  - A `...`: An arbitrary number of dimensions of any sizes.
  - A `str: int` pair (technically it's a slice), combining both `str` and `int` behaviour. (Just a `str` on its own is equivalent to `str: -1`.)
  - A `str: str` pair, in which case the size of the dimension passed at runtime will be bound to _both_ names, and all dimensions with either name must have the same size. (Some people like to use this as a way to associate multiple names with a dimension, for extra documentation purposes.)
  - A `str: ...` pair, in which case the multiple dimensions corresponding to `...` will be bound to the name specified by `str`, and again checked for consistency between arguments.
  - `None`, which when used in conjunction with `is_named` below, indicates a dimension that must _not_ have a name in the sense of [named tensors](https://pytorch.org/docs/stable/named_tensor.html).
  - A `None: int` pair, combining both `None` and `int` behaviour. (Just a `None` on its own is equivalent to `None: -1`.)
  - A `None: str` pair, combining both `None` and `str` behaviour. (That is, it must not have a named dimension, but must be of a size consistent with other uses of the string.)
  - A `typing.Any`: Any size is allowed for this dimension (equivalent to `-1`).
  - Any tuple of the above. For example.`TensorType["batch": ..., "length": 10, "channels", -1]`. If you just want to specify the number of dimensions then use for example `TensorType[-1, -1, -1]` for a three-dimensional tensor.
- The `dtype` argument can be any of:
  - `jax.numpy.float32`, `jax.numpy.float64` etc.
  - `int`, `bool`, `float`, which are converted to their corresponding Jax types. `int` is `int64`, and `float` is specifically interpreted the default dtype of `jax.numpy.ones(())`. See [`_convert_dtype_element`](https://github.com/redwoodresearch/jaxtyping/blob/master/torchtyping/tensor_type.py#L70-L78)
- The `details` argument offers a way to pass an arbitrary number of additional flags that customise and extend `jaxtyping`. One flag is built-in by default. `jaxtyping.is_float` can be used to check that arbitrary floating point types are passed in. (Rather than just a specific one as with e.g. `JaxArray[jax.numpy.float32]`.) For discussion on how to customise `jaxtyping` with your own `details`, see the [further documentation](https://github.com/redwoodresearch/jaxtyping/blob/master/FURTHER-DOCUMENTATION.md#custom-extensions).
- Check multiple things at once by just putting them all together inside a single `[]`. For example `TensorType["batch": ..., "length", "channels", float, is_named]`.

```python
jaxtyping.patch_typeguard()
```

`jaxtyping` integrates with `typeguard` to perform runtime type checking. `jaxtyping.patch_typeguard()` should be called at the global level, and will patch `typeguard` to check `TensorType`s.

This function is safe to run multiple times. (It does nothing after the first run). 

- If using `@typeguard.typechecked`, then `jaxtyping.patch_typeguard()` should be called any time before using `@typeguard.typechecked`. For example you could call it at the start of each file using `jaxtyping`.
- If using `typeguard.importhook.install_import_hook`, then `jaxtyping.patch_typeguard()` should be called any time before defining the functions you want checked. For example you could call `jaxtyping.patch_typeguard()` just once, at the same time as the `typeguard` import hook. (The order of the hook and the patch doesn't matter.)
- If you're not using `typeguard` then `jaxtyping.patch_typeguard()` can be omitted altogether, and `jaxtyping` just used for documentation purposes.

```bash
pytest --jaxtyping-patch-typeguard
```

`jaxtyping` offers a `pytest` plugin to automatically run `jaxtyping.patch_typeguard()` before your tests. `pytest` will automatically discover the plugin, you just need to pass the `--jaxtyping-patch-typeguard` flag to enable it. Packages can then be passed to `typeguard` as normal, either by using `@typeguard.typechecked`, `typeguard`'s import hook, or the `pytest` flag `--typeguard-packages="your_package_here"`.

## Further documentation

See the [further documentation](https://github.com/redwoodresearch/jaxtyping/blob/master/FURTHER-DOCUMENTATION.md) for:

- FAQ;
  - Including `flake8` and `mypy` compatibility;
- How to write custom extensions to `jaxtyping`;
- Resources and links to other libraries and materials on this topic;
- More examples.
