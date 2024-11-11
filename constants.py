import jax.numpy as jnp

default_dtype = jnp.float32
EVEN_PARITY = 1 # a mnemonic to remember this is: "multiplying by 1 doesn't change the sign"
ODD_PARITY = -1

NUM_PARITY_DIMS = 2
EVEN_PARITY_IDX = 0
ODD_PARITY_IDX = 1

PARITY_IDXS = [EVEN_PARITY_IDX, ODD_PARITY_IDX]