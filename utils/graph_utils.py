import jax.numpy as jnp
from typing import Tuple
import numpy as np


# from https://chatgpt.com/share/6726811b-96a0-800e-af41-684b211f59b6
# TODO: support periodic boundary conditions. the graphs made here are NOT periodic (we'd need to take in the lattice paramters for periodic support)
def radius_graph(
    positions: jnp.ndarray, radius: float
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Returns the indices of the senders and receivers of a radius graph.

    Args:
        positions: The positions of the nodes.
        radius: The radius of the graph.

    Returns:
        senders: The indices of the senders of the graph.
        receivers: The indices of the receivers of the graph.
    """
    # Compute pairwise squared distances
    diffs = (
        positions[:, None, :] - positions[None, :, :]
    )  # Shape: (N, N, D) where N is the number of nodes, and D is the dimensionality
    dists_squared = jnp.sum(diffs**2, axis=-1)  # Shape: (N, N)

    # Create a mask for distances within the radius (excluding self-distances)
    mask = (dists_squared <= radius**2) & (dists_squared > 0)

    # Extract sender and receiver indices
    senders, receivers = jnp.where(mask)

    return senders, receivers


def to_graph(
    positions: list[float], cutoff_radius: int, nodes_have_self_connections: bool
) -> tuple[np.ndarray, np.ndarray]:
    """
    Converts a point cloud into a graph based on a cutoff radius.

    Parameters:
    - positions (list[float]): A list of positions, where each position is a list of coordinates.
    - cutoff_radius (int): The maximum distance between nodes to consider an edge.
    - nodes_have_self_connections (bool): If True, nodes will have edges to themselves.

    Returns:
    - source_indices (np.ndarray): Array of source node indices for edges.
    - target_indices (np.ndarray): Array of target node indices for edges.
    """
    # Convert positions to a NumPy array for vectorized operations
    positions = np.array(positions)
    # num_nodes = positions.shape[0]

    # Calculate pairwise distance matrix
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    distances = np.linalg.norm(diff, axis=2)

    # Determine adjacency based on cutoff_radius
    adjacency = distances <= cutoff_radius

    # Exclude self-connections if specified
    if not nodes_have_self_connections:
        np.fill_diagonal(adjacency, False)

    # Get the indices of edges
    source_indices, target_indices = np.nonzero(adjacency)

    return source_indices, target_indices
