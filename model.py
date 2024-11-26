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
        self.radius = 11

        # first layer
        self.layer1 = Layer(self.starting_irreps_id, "5x0e + 5x1o")
        self.activation_layer1 = ActivationLayer("GELU", "5x0e + 5x1o")
        self.layer2 = Layer("5x0e + 5x1o", "5x0e + 5x1o")
        self.activation_layer2 = ActivationLayer("GELU", "5x0e + 5x1o")
        self.layer3 = Layer("5x0e + 5x1o", "5x0e")
        self.activation_layer3 = ActivationLayer("GELU", "5x0e + 5x1o")

        # intermediate layers
        # self.layer2 = Layer("5x0e + 5x1o", "5x0e + 5x1o")
        # self.layer3 = Layer("5x0e + 5x1o", "5x0e + 5x1o")

        # output layer
        num_scalar_features = 5  # since the output of layer3 is 5x
        self.output_mlp = torch.nn.Linear(
            num_scalar_features, num_classes, dtype=default_dtype
        )
        self.softmax = torch.nn.Softmax(dim=-1)

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
        x = self.activation_layer1(x)
        x = self.layer2(x, edge_index, positions)
        x = self.activation_layer2(x)
        x = self.layer3(x, edge_index, positions)
        x = self.activation_layer3(x)
        # x = self.layer2(x, edge_index, positions)
        # x = self.layer3(x, edge_index, positions)

        # now pool the features on each node to generate the final output irreps
        pooled_feats = avg_irreps_with_same_id(x)
        scalar_feats = [irrep.data for irrep in pooled_feats.get_irreps_by_id("0e")]
        return self.softmax(self.output_mlp(torch.cat(scalar_feats)))


# IMPORTANT: LinearLayer are the weights for an individual node. you re-use it for each different node in the graph
# in general, if the forward function returns list[Irreps], it processes on all the nodes in the graph. if it returns Irreps, it processes on a single node
class LinearLayer(torch.nn.Module):
    # How this linear layer works:
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
        self.input_irrep_id_cnt, _sorted_input_ids = Irreps.count_num_irreps(
            input_irreps_id
        )
        self.output_irrep_id_cnt, self.sorted_output_ids = Irreps.count_num_irreps(
            output_irreps_id
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
            num_output_irreps_of_id = self.output_irrep_id_cnt[irrep_id]
            num_input_irreps_of_id = self.input_irrep_id_cnt[irrep_id]
            num_weights_for_l = 2 * l + 1

            # each one of the same output irreps for this id will be multiplied by a linear combinations of the same input irreps. so loop this for loop |num_output_coefficients| times
            for _ in range(num_output_irreps_of_id):
                # for each output irrep, we need to multiply each of the input irreps by a different weight. these are the weights for the input irreps for this output irrep
                for _ in range(num_input_irreps_of_id):
                    self.weights.append(nn.Parameter(torch.randn(num_weights_for_l)))

        # 3) add biases to 0e irreps (we can only add biases to these irreps cause they are invariant - adding it to other irreps will mess up the equivariance of the system)
        self.use_bias = use_bias
        if self.use_bias:
            # we can only apply biases to even scalar outputs (as they are invariant)
            num_even_scalar_outputs = self.output_irrep_id_cnt["0e"]
            # we just need a single bias for each output 0e irrep (irrespective of the number of inputs. since adding a bias for each input is the same as just adding one for the output)
            self.biases = nn.Parameter(torch.randn(num_even_scalar_outputs))

    def forward(self, x: Irreps) -> Irreps:
        output_irreps: list[Irrep] = []
        weight_idx = 0
        for i in range(len(self.sorted_output_ids)):
            out_irrep_id, l, parity = self.sorted_output_ids[i]

            # this is the number of times this irrep is produced in the output:
            num_output_irreps_of_id = self.output_irrep_id_cnt[out_irrep_id]

            # we need to generate this many output coefficients. TODO(curtis): can we reduce this to just one for loop?
            for _ in range(num_output_irreps_of_id):
                data_out = torch.zeros(l * 2 + 1, dtype=default_dtype)

                # loop through all of the input irreps for this output irrep
                for irrep in x.get_irreps_by_id(out_irrep_id):
                    irrep_weights = self.weights[weight_idx]
                    weight_idx += 1
                    data_out += irrep.data * irrep_weights
                output_irreps.append(Irrep(l, parity, data_out))

        # now add the biases
        if self.use_bias:
            bias_idx = 0
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
        # TODO(curtis): maybe irreps_id_after_tensor_product is wrong?
        irreps_id_after_tensor_product = self._get_irreps_id_after_tensor_product(
            input_irreps_id
        )
        self.after_tensor_prod = LinearLayer(
            irreps_id_after_tensor_product, output_irreps_id
        )
        # self.addition = nn.Parameter(torch.randn(3))
        # self.addition2 = nn.Parameter(torch.randn(1), requires_grad=True)

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
    ) -> list[Irreps]:
        """
        x: Node features [num_nodes, input_dim]
        edge_index: Edge indices [2, num_edges]
        positions: Node positions [num_nodes, 3]

        outputs a list of Irreps (one for each node)
        """

        # Compute relative positions.
        src_nodes, dest_nodes = edge_index  # Senders (source), Receivers (target)
        relative_positions = (
            positions[dest_nodes] - positions[src_nodes]
        )  # [num_edges, 3]
        # print("relative positions", relative_positions)

        # Compute spherical harmonics.
        all_sh_feats = map_3d_feats_to_spherical_harmonics_repr(
            relative_positions, self.sh_lmax
        )  # [num_edges, sh_dim]
        # NOTE: we can multiply the output of sh via scalar weights (much like the input to the allegro model)

        new_edge_feats: list[Irreps] = []
        for (
            idx,
            sh,
        ) in enumerate(all_sh_feats):
            dest_node: int = dest_nodes[idx]
            dest_node_feat = x[dest_node]
            # print("sh", sh)
            # sh.data()[1].data += self.addition
            # sh.data()[0].data += self.addition2

            # Compute tensor product
            tensor_product = dest_node_feat.tensor_product(
                sh, compute_up_to_l=self.sh_lmax, norm_type="component"
            )
            # tensor_product.data()[0].data += self.addition2

            # print("tensor_product", tensor_product)
            # tensor_product.avg_irreps_of_same_id()
            weighted_tensor_product: Irreps = self.after_tensor_prod(tensor_product)
            # weighted_tensor_product.data()[0].data += self.addition2
            new_edge_feats.append(weighted_tensor_product)

        # now that we have the new edge features, we aggregate them to get the new features for each node
        # incoming_edge_features_for_each_node =
        new_node_features: list[Irreps] = []
        for node_idx in range(len(x)):
            incoming_edge_features = []
            for incoming_edge_idx, dest_node_idx in enumerate(dest_nodes):
                if dest_node_idx == node_idx:
                    incoming_edge_features.append(new_edge_feats[incoming_edge_idx])
            aggregated_incoming_edge_features = avg_irreps_with_same_id(
                incoming_edge_features
            )
            # aggregated_incoming_edge_features.data()[0].data += self.addition2
            # aggregated_incoming_edge_features.data()[0].data = torch.zeros(1)
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


# How this works:
# 1) given input irreps (e.g. 1x0e + 1x1o), we count the number of irreps (there are 2)
# 2) we then create a linear layer that takes in the input irreps and creates the same number of scalar irreps (since there are 2 irreps, we create 2 scalar irreps)
# 3) we then put these scalar irreps into the activation function and multiply the result by the original irreps
# 4) return the new irreps!
#
# NOTE: I do NOT treat 0o as an invariant scalar. so 0o features are multiplied by the output of activation functions of 0e features
# e3simple doesn't support this, but if you want to use 0o features as inputs to activation functions, you need to be careful and respect the OUTPUT parity of the activation function.
# see the Activation class in e3nn for more details. The parity of the activation function just changes the parity of the output irrep (I think after multiplying by the result of act(scalar_irrep)):
#  a1, a2 = act(x), act(-x)
#  if (a1 - a2).abs().max() < 1e-5:
#      p_act = 1
#  elif (a1 + a2).abs().max() < 1e-5:
#      p_act = -1
#  else:
#      p_act = 0
#
#
#  p_out = p_act if p_in == -1 else p_in
#  irreps_out.append((mul, (0, p_out)))
class ActivationLayer(nn.Module):
    linearLayer: LinearLayer

    def __init__(
        self,
        activation_fn_str: str,
        input_irreps_id: str,
    ):
        super().__init__()

        irrep_id_cnt, _sorted_ids = Irreps.count_num_irreps(input_irreps_id)
        assert (
            irrep_id_cnt["0e"] > 0
        ), "You must have at least one 0e irrep to use the activation layer. This is because we need scalar features to pass into the activation function. Nonscalar features cannot be fed into the activation function since they are not equivariant!"
        num_irreps = sum(irrep_id_cnt.values())
        output_irreps_id = f"{num_irreps}x0e"  # e.g. for 1x0e+1x1o irreps, we want the linear layer to output 2 scalars. we will take these two scalars and feed it into a gate which will individually scale the 1x0e and 1x1o irrep of the original irreps

        self.linear_layer = LinearLayer(input_irreps_id, output_irreps_id)

        if activation_fn_str == "ReLU":
            self.activation_fn = nn.ReLU()
        elif activation_fn_str == "GELU":
            self.activation_fn = nn.GELU()
        else:
            raise ValueError(f"activation_fn_str {activation_fn_str} is not supported")

        self.activation_fn_str = activation_fn_str

    def forward(self, x: list[Irreps]) -> list[Irreps]:
        out = []
        for node_irreps in x:  # for each node's irreps
            # each scalar irrep will scale the original irreps
            linear_output: Irreps = self.linear_layer(node_irreps)
            scalar_irreps = linear_output.get_irreps_by_id("0e")

            out_irreps = []  # these are the output irreps for this node
            for i in range(len(node_irreps.irreps)):
                irrep = node_irreps.irreps[i]
                scalar_irrep = scalar_irreps[i]

                new_irrep_data = self.activation_fn(scalar_irrep.data) * irrep.data
                out_irreps.append(Irrep(irrep.l, irrep.parity, new_irrep_data))

            out.append(Irreps(out_irreps))
        return out
