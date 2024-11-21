import torch
import torch.nn as nn
import torch.nn.functional as F
from spherical_harmonics import map_3d_feats_to_spherical_harmonics_repr
from torch_scatter import scatter_add
from torch_geometric.nn import MessagePassing


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 1)

    def forward(self, data):
        return self.linear(data.x)

class Layer(MessagePassing):
    def __init__(self, input_dim, target_dim, denominator, sh_lmax=3):
        super(Layer, self).__init__(aggr='add')  # Use 'add' aggregation.
        self.denominator = denominator
        self.sh_lmax = sh_lmax

        # Define linear layers.
        self.linear_pre = nn.Linear(input_dim, target_dim)
        self.linear_post = nn.Linear(target_dim, target_dim)
        self.shortcut = nn.Linear(input_dim, target_dim)

    def forward(self, x, edge_index, positions):
        """
        x: Node features [num_nodes, input_dim]
        edge_index: Edge indices [2, num_edges]
        positions: Node positions [num_nodes, 3]
        """
        #TODO: do we need to make a graph here?

        # Compute relative positions.
        row, col = edge_index  # Senders (source), Receivers (target)
        rel_pos = positions[col] - positions[row]  # [num_edges, 3]

        # Compute spherical harmonics.
        sh = map_3d_feats_to_spherical_harmonics_repr(rel_pos, self.sh_lmax)  # [num_edges, sh_dim]

        sender_features = x[row]  # [num_edges, input_dim]

        # Compute tensor product (outer product) and flatten.
        tensor_prod = torch.einsum('ef, eg -> efg', sender_features, sh)
        tensor_prod = tensor_prod.reshape(tensor_prod.size(0), -1)  # [num_edges, input_dim * sh_dim]

        # Concatenate sender features and tensor product.
        edge_features = torch.cat([sender_features, tensor_prod], dim=1)  # [num_edges, edge_feat_dim]

        # Proceed with message passing.
        out = self.propagate(edge_index, x=x, edge_features=edge_features)

        return out

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
