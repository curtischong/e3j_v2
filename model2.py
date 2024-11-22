from collections import Counter, defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
from geometric_utils import avg_irreps_with_same_id, to_graph
from irrep import Irrep, Irreps
from spherical_harmonics import map_3d_feats_to_spherical_harmonics_repr
import numpy as np


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 1)
        self.layer1 = Layer(20, 20, 1)
        self.radius = 1.1

    def forward(self, positions):
        num_nodes = len(positions)
        starting_irreps = []
        for _ in range(num_nodes):
            starting_irreps.append(Irreps.from_id("1x0e", [torch.ones(1)]))

        edge_index = to_graph(
            positions, cutoff_radius=1.5, nodes_have_self_connections=False
        )  # make nodes NOT have self connections since that messes up with the relative positioning when we're calculating the spherical harmonics (the features need to be points on a sphere, but a distance of 0 cannot be normalized to a point on the sphere (divide by 0))

        # perform message passing and get new irreps
        x = self.layer1(starting_irreps, edge_index, positions)
        # now make each node go through a linear layer

        return x


class LinearLayer(torch.nn.Module):
    # unfortunately, we cannot determine input_irreps_id at runtime since we need to init the linear layer here first
    # if it's possible to delay it until runtim, that would be great! because then we could just pass in the irreps to the layer (no need to specify if of input irreps)
    def __init__(self, input_irreps_id: str, output_irreps_id: str):
        num_input_coefficients = 0
        self.sorted_input_ids = []
        self.unique_input_ids = defaultdict(int)
        # Note: I think it's okay if all of the input irreps have num_irreps=1, because we're still multiplying each irrep by a weight
        # we can also use this linear layer to filter out representations we don't care about (e.g. the final layer for prediction)
        # Irreps.parse_id returns irreps in sorted order. so we can depend on this order when assigning weights
        for _irrep_def, num_irreps, l, parity in Irreps.parse_id(input_irreps_id):
            self.sorted_input_ids.append((l, parity))
            num_input_coefficients += num_irreps * (2 * l + 1)
            self.unique_input_ids[Irrep.to_id(l, parity)] += num_irreps

        num_output_coefficients = 0
        self.unique_output_ids = defaultdict(int)
        for _irrep_def, num_irreps, l, _parity in Irreps.parse_id(output_irreps_id):
            num_output_coefficients += num_irreps * (2 * l + 1)
            self.unique_output_ids[Irrep.to_id(l, parity)] += num_irreps

        for unique_output_id in self.unique_output_ids:
            assert (
                unique_output_id in self.unique_input_ids
            ), f"output irrep {unique_output_id} is not in the input irreps. We cannot create this output irrep because it's not in the input irreps. Maybe do a tensor product before putting it into this linear layer to get those desired irreps?"

        num_weights = num_input_coefficients * num_output_coefficients
        self.weights = nn.Parameter(torch.randn(num_weights, requires_grad=True))

    def forward(self, x: Irreps):
        for irrep in x.irreps:
            irrep_id = irrep.id()
            if irrep_id not in self.unique_output_ids:
                continue
            num_output_coefficients = self.unique_output_ids[irrep_id]

            # since the input ids are sorted in the same order,


class Layer(torch.nn.Module):
    def __init__(self, input_dim: int, target_dim: int, sh_lmax=2):
        super(Layer, self).__init__()
        self.sh_lmax = sh_lmax

        # Define linear layers.
        self.linear_pre = nn.Linear(input_dim, target_dim)
        self.linear_post = nn.Linear(target_dim, target_dim)
        self.shortcut = nn.Linear(input_dim, target_dim)

    def forward(
        self, x: list[Irreps], edge_index: tuple[np.ndarray, np.ndarray], positions
    ):
        """
        x: Node features [num_nodes, input_dim]
        edge_index: Edge indices [2, num_edges]
        positions: Node positions [num_nodes, 3]
        """

        # Compute relative positions.
        src_nodes, dest_nodes = edge_index  # Senders (source), Receivers (target)
        relative_positions = (
            positions[dest_nodes] - positions[src_nodes]
        )  # [num_edges, 3]

        # Compute spherical harmonics.
        sh = map_3d_feats_to_spherical_harmonics_repr(
            relative_positions, self.sh_lmax
        )  # [num_edges, sh_dim]
        # NOTE: we can multiply the output of sh via scalar weights (much like the input to the allegro model)

        new_edge_feats: list[Irreps] = []
        for (
            idx,
            sh,
        ) in enumerate(sh):
            dest_node: int = dest_nodes[idx]
            dest_node_feat = x[dest_node]

            # Compute tensor product
            tensor_product = dest_node_feat.tensor_product(
                sh, compute_up_to_l=self.sh_lmax
            )
            tensor_product.avg_irreps_of_same_id()
            new_edge_feats.append(tensor_product)

        # now that we have the new edge features, we aggregate them to get the new features for each node
        # incoming_edge_features_for_each_node =
        new_node_features = []
        for node_idx in range(len(x)):
            incoming_edge_features = []
            for incoming_edge_idx, dest_node_idx in enumerate(dest_nodes):
                if dest_node_idx == node_idx:
                    incoming_edge_features.append(new_edge_feats[incoming_edge_idx])
                    continue
            aggregated_incoming_edge_features = avg_irreps_with_same_id(
                incoming_edge_features
            )
            new_node_features.append(aggregated_incoming_edge_features)

        return new_node_features

    def message(self, edge_features):
        # Messages are the edge features.
        return edge_features

    def update(self, aggr_out, x):
        """
        aggr_out: Aggregated messages [num_nodes, edge_feat_dim]
        x: Node features [num_nodes, input_dim]
        """
        # Node update as per the original code.
        node_feats = x / self.denominator
        node_feats = self.linear_pre(node_feats)
        node_feats = F.relu(node_feats)
        node_feats = self.linear_post(node_feats)
        shortcut = self.shortcut(x)
        return shortcut + node_feats
