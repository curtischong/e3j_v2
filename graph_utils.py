import jax.numpy as jnp
from typing import Tuple

# from https://chatgpt.com/share/6726811b-96a0-800e-af41-684b211f59b6
# TODO: support periodic boundary conditions. the graphs made here are NOT periodic (we'd need to take in the lattice paramters for periodic support)
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
    diffs = positions[:, None, :] - positions[None, :, :]  # Shape: (N, N, D) where N is the number of nodes, and D is the dimensionality
    dists_squared = jnp.sum(diffs ** 2, axis=-1)            # Shape: (N, N)

    # Create a mask for distances within the radius (excluding self-distances)
    mask = (dists_squared <= radius ** 2) & (dists_squared > 0)

    # Extract sender and receiver indices
    senders, receivers = jnp.where(mask)

    return senders, receivers
