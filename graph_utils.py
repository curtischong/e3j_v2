import jax.numpy as jnp
from typing import Tuple

def radius_graph(positions: jnp.ndarray, radius: float) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Returns the indices of the senders and receivers of a radius graph.

    Args:
        positions: The positions of the nodes.
        radius: The radius of the graph.

    Returns:
        senders: The indices of the senders of the graph.
        receivers: The indices of the receivers of the graph.
    """
    # Compute pairwise squared distances
    diffs = positions[:, None, :] - positions[None, :, :]  # Shape: (N, N, D)
    dists_squared = jnp.sum(diffs ** 2, axis=-1)            # Shape: (N, N)

    # Create a mask for distances within the radius (excluding self-distances)
    mask = (dists_squared <= radius ** 2) & (dists_squared > 0)

    # Extract sender and receiver indices
    senders, receivers = jnp.where(mask)

    return senders, receivers
