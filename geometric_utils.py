# for a single set of poin clouds (one training example), convert it into a graph
import numpy as np

from irrep import Irreps


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


# used when we want to aggregate all of the irreps coming from neighbouring nodes (right after when the messages are passed)
def avg_irreps_with_same_id(irreps_list: list[Irreps]) -> Irreps:
    ids = [irreps.id() for irreps in irreps_list]
    assert (
        len(set(ids)) == 1
    ), f"All ids for irreps in irreps_list MUST be the same (so we can avg them). ids: {ids}"

    # the first irreps will be where we store the averaged irreps
    summed_data = irreps_list[0].data()
    for irreps in irreps_list[1:]:
        for idx, irrep in enumerate(irreps.irreps):
            summed_data[idx] += irrep.data

    for i in range(len(summed_data)):
        summed_data[i] /= len(irreps_list)
    return Irreps.from_id(irreps_list[0].id(), summed_data)
