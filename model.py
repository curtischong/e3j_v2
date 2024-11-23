from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
from geometric_utils import avg_irreps_with_same_id, to_graph
from irrep import Irrep, Irreps
from spherical_harmonics import map_3d_feats_to_spherical_harmonics_repr
import numpy as np
from constants import default_dtype

from utils.dummy_data_utils import create_irreps_with_dummy_data


class Model(torch.nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.starting_irreps_id = "1x0e"  # each node starts with a dummy 1x0e irrep
        self.layer1 = Layer(self.starting_irreps_id, "1x0e + 1x1o")
        self.radius = 1.1
        num_scalar_features = 1  # since the output of layer1 is 1x
        self.output_mlp = torch.nn.Linear(
            num_scalar_features, num_classes, dtype=default_dtype
        )

    def forward(self, positions):
        num_nodes = len(positions)
        starting_irreps = []
        for _ in range(num_nodes):
            starting_irreps.append(
                create_irreps_with_dummy_data(self.starting_irreps_id)
            )

        edge_index = to_graph(
            positions, cutoff_radius=1.5, nodes_have_self_connections=False
        )  # make nodes NOT have self connections since that messes up with the relative positioning when we're calculating the spherical harmonics (the features need to be points on a sphere, but a distance of 0 cannot be normalized to a point on the sphere (divide by 0))

        # perform message passing and get new irreps
        x = self.layer1(starting_irreps, edge_index, positions)

        # now pool the features
        pooled_feats = avg_irreps_with_same_id(x)
        scalar_feats = pooled_feats.data()[0].data
        return self.output_mlp(scalar_feats)


class LinearLayer(torch.nn.Module):
    # How ths linear layer works:
    # for each irrep of the same id, we take a weighted sum of these irreps to create output irreps OF THE SAME ID. Irreps of different IDs will NOT be combined together
    # (e.g. 1o_0*w0 + 1o_1*w1 + 1o_2*w2 -> 1o_out)
    # Note: you can specify the number of output irreps you want. so if you want 3 1o irreps, we do the above 3 times. each time with diff weights:
    # 1o_0*w0 + 1o_1*w1 + 1o_2*w2 -> 1o_out1
    # 1o_0*w3 + 1o_1*w4 + 1o_2*w5 -> 1o_out2
    # 1o_0*w6 + 1o_1*w7 + 1o_2*w8 -> 1o_out3
    #
    # One useful place to use the linear layer is right after the tensor product. Since it expands the number of irreps. we can use this to "combine" the irreps back down to a manageable number of coefficients
    # You can also use this if you want to mix features in each irrep of the same id (e.g. the input/output ids are the same)
    # You can also use this if you want to increase/expand the irrep types at any point in the network (e.g. create 5 times more 1o irreps)
    #
    # unfortunately, we cannot determine input_irreps_id at runtime since we need to init the linear layer here first (so it must be passed in as a parameter)
    # if it's possible to delay it until runtime, that would be great! because then we could just pass in the irreps to the layer (no need to specify if of input irreps)
    #
    # when we use bias, we also add a bias term to all the 0e irreps
    def __init__(
        self, input_irreps_id: str, output_irreps_id: str, use_bias: bool = True
    ):
        super().__init__()

        # 1) In the init function, we need to determine the number of weights to create each output irrep based on the input irreps
        # In this step, we are counting the number of weights we need to initialize
        # Note: I think it's okay if all of the input irreps have num_irreps=1, because we're still multiplying each irrep by a weight
        # we can also use this linear layer to filter out representations we don't care about (e.g. the final layer for prediction)
        # Irreps.parse_id returns irreps in sorted order. so we can depend on this order when assigning weights
        self.input_irrep_id_cnt, num_input_coefficients, _sorted_input_ids = (
            self._count_num_irreps(input_irreps_id)
        )
        self.output_irrep_id_cnt, num_output_coefficients, self.sorted_output_ids = (
            self._count_num_irreps(output_irreps_id)
        )

        for unique_output_id in self.output_irrep_id_cnt:
            assert (
                unique_output_id in self.input_irrep_id_cnt
            ), f"output irrep {unique_output_id} is not in the input irreps. We cannot create this output irrep because it's not in the input irreps. Maybe do a tensor product before putting it into this linear layer to get those desired irreps?"

        # 2) now that we know the number of input and output irreps, we can create the weights that transforms the input irreps to the output irreps
        # How the below code works:
        # for each of the input irreps of the same id, we need to create |num_irreps_of_same_id|*(number of coefficients for the id's l) for each output irrep
        self.weights = nn.ParameterList()

        for irrep_id, l, parity in self.sorted_output_ids:
            num_output_coefficients = self.output_irrep_id_cnt[irrep_id]
            num_input_coefficients = self.input_irrep_id_cnt[irrep_id]

            # example to teach the reasoning for the num_weights:
            # 1o_1*w1 + 1o_2*w2 -> 1o_out
            # in the above example, we have two 1o input irreps. since both 1o irrep has 3 coefficients, we need 2*3 = 6 weights to transform the input irreps to the output irreps
            num_weights_for_l = 2 * l + 1
            num_weights = num_input_coefficients * num_weights_for_l

            # each one of the same output irreps for this id will be multiplied by a linear combinations of the same input irreps. so loop this for loop |num_output_coefficients| times
            for _ in range(num_output_coefficients):
                self.weights.append(
                    nn.Parameter(torch.randn(num_weights, requires_grad=True))
                )

        self.use_bias = use_bias
        if self.use_bias:
            # we can only apply biases to even scalar outputs (as they are invariant)
            num_even_scalar_outputs = self.output_irrep_id_cnt["0e"]
            # we just need a single bias for each output 0e irrep (irrespective of the number of inputs. since adding a bias for each input is the same as just adding one for the output)
            self.biases = nn.Parameter(torch.randn(num_even_scalar_outputs))

    def _count_num_irreps(self, irreps_id: str) -> int:
        num_coefficients = 0
        irrep_id_cnt = defaultdict(int)
        sorted_ids = []

        for _irrep_def, num_irreps, l, parity in Irreps.parse_id(irreps_id):
            irrep_id = Irrep.to_id(l, parity)
            sorted_ids.append((irrep_id, l, parity))
            num_coefficients += num_irreps * (2 * l + 1)
            irrep_id_cnt[irrep_id] += num_irreps
        return (irrep_id_cnt, num_coefficients, sorted_ids)

    def forward(self, x: Irreps) -> Irreps:
        cur_weight_idx = 0
        output_irreps: list[Irrep] = []
        for i in range(len(self.sorted_output_ids)):
            irrep_id, l, parity = self.sorted_output_ids[i]
            data_out = torch.zeros(l * 2 + 1, dtype=default_dtype)
            for irrep in x.get_irreps_by_id(irrep_id):
                data_out += irrep.data * self.weights[cur_weight_idx]
                cur_weight_idx += 1
            output_irreps.append(Irrep(l, parity, data_out))

        # now add the biases
        bias_idx = 0
        if self.use_bias:
            for irrep in output_irreps:
                if irrep.id() == "0e":
                    irrep.data += self.biases[bias_idx]
                    bias_idx += 1
        return Irreps(output_irreps)


class Layer(torch.nn.Module):
    def __init__(self, input_irreps_id: str, output_irreps_id: str, sh_lmax=1):
        super().__init__()
        self.sh_lmax = sh_lmax

        # Define linear layers.
        irreps_id_after_tensor_product = self._get_irreps_id_after_tensor_product(
            input_irreps_id
        )
        self.after_tensor_prod = LinearLayer(
            irreps_id_after_tensor_product, output_irreps_id
        )

    def _get_irreps_id_after_tensor_product(self, input_irreps_id: str) -> str:
        # perform a dummy tensor product to get the irreps_id going into the linear layer after
        # the tensor product layer
        sh_dummy_irreps = map_3d_feats_to_spherical_harmonics_repr(
            torch.tensor([[1.0, 0.0, 0.0]]), self.sh_lmax
        )[0]
        input_dummy_irreps = create_irreps_with_dummy_data(input_irreps_id)
        return input_dummy_irreps.tensor_product(sh_dummy_irreps).id()

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
            # tensor_product.avg_irreps_of_same_id()
            weighted_tensor_product = self.after_tensor_prod(tensor_product)
            new_edge_feats.append(weighted_tensor_product)

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


# which linear feature to use for the activation function?
def activation(irreps: Irreps, activation_fn: str) -> Irreps:
    data = irreps.data_flattened()
    if activation_fn == "relu":
