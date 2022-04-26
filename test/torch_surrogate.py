import jax.numpy as jnp
import jaxtyping


TensorType = jaxtyping.JaxArray

def rand(*shape):
    return jnp.zeros(shape)

tensor = jnp.array
sparse_coo = NotImplemented
strided = NotImplemented
