import torch
import torch.nn as nn
import torch.nn.functional as F
from geometric_utils import avg_irreps_with_same_id, to_graph
from irrep import Irreps
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

        edge_index = to_graph(positions, cutoff_radius=1.5, nodes_have_self_connections=False) # make nodes NOT have self connections since that messes up with the relative positioning when we're calculating the spherical harmonics (the features need to be points on a sphere, but a distance of 0 cannot be normalized to a point on the sphere (divide by 0))

        x= self.layer1(starting_irreps, edge_index, positions)
        return x

class Layer(torch.nn.Module):
    def __init__(self, input_dim: int, target_dim: int, denominator: int, sh_lmax=2):
        super(Layer, self).__init__()
        self.denominator = denominator
        self.sh_lmax = sh_lmax

        # Define linear layers.
        self.linear_pre = nn.Linear(input_dim, target_dim)
        self.linear_post = nn.Linear(target_dim, target_dim)
        self.shortcut = nn.Linear(input_dim, target_dim)

    def forward(self, x: list[Irreps], edge_index: tuple[np.ndarray, np.ndarray], positions):
        """
        x: Node features [num_nodes, input_dim]
        edge_index: Edge indices [2, num_edges]
        positions: Node positions [num_nodes, 3]
        """

        # Compute relative positions.
        src_nodes, dest_nodes = edge_index  # Senders (source), Receivers (target)
        relative_positions = positions[dest_nodes] - positions[src_nodes]  # [num_edges, 3]

        # Compute spherical harmonics.
        sh = map_3d_feats_to_spherical_harmonics_repr(relative_positions, self.sh_lmax)  # [num_edges, sh_dim]

        new_edge_feats: list[Irreps] = []
        for idx, sh, in enumerate(sh):
            dest_node: int = dest_nodes[idx]
            dest_node_feat = x[dest_node]

            # Compute tensor product
            tensor_product = dest_node_feat.tensor_product(sh, compute_up_to_l=self.sh_lmax)
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
            aggregated_incoming_edge_features = avg_irreps_with_same_id(incoming_edge_features)
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
