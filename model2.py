import torch
import torch.nn as nn
import torch.nn.functional as F
from geometric_utils import to_graph
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

        edge_index = to_graph(positions, cutoff_radius=1.5, nodes_have_self_connections=True)

        # todo: create graph with edges connections here
        y = self.layer1(starting_irreps, edge_index, positions)
        return y

class Layer(torch.nn.Module):
    def __init__(self, input_dim: int, target_dim: int, denominator: int, sh_lmax=3):
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

        # new_dest_node_feats = [torch.zeros(self.sh_lmax*2 + 1)]*len(x) # init these feats to 0 since we're doing an aggregation
        new_edge_feats = []
        for idx, sh, in enumerate(sh):
            dest_node: int = dest_nodes[idx]
            dest_node_feat = x[dest_node]

            # Compute tensor product
            tensor_product = dest_node_feat.tensor_product(sh, compute_up_to_l=self.sh_lmax)
            tensor_product_consolidated_feats = tensor_product.avg_irreps_of_same_id()
            new_edge_feats.append(tensor_product_consolidated_feats)

        # now that we have the new edge features, we aggregate them to get the new features for each node



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
