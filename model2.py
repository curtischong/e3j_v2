import torch
import torch.nn as nn
import torch.nn.functional as F
from irrep import Irreps
from spherical_harmonics import map_3d_feats_to_spherical_harmonics_repr
from torch_scatter import scatter_add
from torch_geometric.nn import MessagePassing

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree
from torch_geometric.transforms import RadiusGraph

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 1)
        self.layer1 = Layer(20, 20, 1)
        self.radius = 1.1
        self.radius_graph = RadiusGraph(self.radius, loop=True, max_num_neighbors=1000)

    def forward(self, data):
        num_nodes = len(data.x[0])
        # starting_irreps = Irreps.from_id(f"{num_nodes}x0e", [torch.ones(1) for _ in range(num_nodes)])

        graph = self.radius_graph(data)

        # todo: create graph with edges connections here
        y = self.layer1(data, graph.edge_index, graph.pos)
        return y

class Layer(MessagePassing):
    def __init__(self, input_dim: int, target_dim: int, denominator: int, sh_lmax=3):
        super(Layer, self).__init__(aggr='add')  # Use 'add' aggregation.
        self.denominator = denominator
        self.sh_lmax = sh_lmax

        # Define linear layers.
        self.linear_pre = nn.Linear(input_dim, target_dim)
        self.linear_post = nn.Linear(target_dim, target_dim)
        self.shortcut = nn.Linear(input_dim, target_dim)

    def forward(self, x: list[Irreps], edge_index, positions):
        """
        x: Node features [num_nodes, input_dim]
        edge_index: Edge indices [2, num_edges]
        positions: Node positions [num_nodes, 3]
        """

        # Compute relative positions.
        row, col = edge_index  # Senders (source), Receivers (target)
        rel_pos = positions[col] - positions[row]  # [num_edges, 3]

        # Compute spherical harmonics.
        sh = map_3d_feats_to_spherical_harmonics_repr(rel_pos, self.sh_lmax)  # [num_edges, sh_dim]

        sender_features = []
        for idx in row:
            sender_features.append(x[idx])  # [num_edges, input_dim]

        # Compute tensor product
        tensor_prod = sender_features.tensor_product(sh)

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
