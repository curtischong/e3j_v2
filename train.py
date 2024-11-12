# Adapted from https://github.com/e3nn/e3nn-jax/blob/main/examples/tetris_point.py
#  * removed scalar non-linearity for now
#  * added exports to .bin and .mp files for weights
import struct
import time

import numpy as np
import flax
import flax.serialization
import jax
import jax.numpy as jnp
import jraph
import optax

from constants import NUM_PARITY_DIMS, default_dtype
from graph_utils import radius_graph
from irrep import Irrep
from spherical_harmonics import map_3d_feats_to_spherical_harmonics_repr
from tensor_product import tensor_product_v1
from jaxtyping import Array, Float


shapes = [
    # [[0, 0, 0], [0, 0, 1], [1, 0, 0], [1, 1, 0]],  # chiral_shape_1 # curtis: chiral_shape_1 and chiral_shape_2 are the same except I think chiral_shape_2 is reflected. Either way, it makes the output irreps harder to predict (need a o1 output to differentiate between the two tetrises.
    # Since I'm lazy, and want this to be as simple as possible, I will just have one chiral shape
    [[1, 1, 1], [1, 1, 2], [2, 1, 1], [2, 0, 1]],  # chiral_shape_2
    [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]],  # square
    [[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 0, 3]],  # line
    [[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]],  # corner
    [[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 1, 0]],  # L
    [[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 1, 1]],  # T # curtis: how is this a T???? doens't it need 5 points?
    [[0, 0, 0], [1, 0, 0], [1, 1, 0], [2, 1, 0]],  # zigzag
]
num_classes = len(shapes)
shapes = jnp.array(shapes, dtype=default_dtype)

def tetris() -> jraph.GraphsTuple:
    labels = jnp.arange(num_classes)

    graphs = []

    for i in range(len(shapes)):
        pos = shapes[i]
        label = labels[i]
        # print("pos", pos.shape)
        # senders, receivers = radius_graph(pos, 1.1)
        senders, receivers = radius_graph(pos, 10) # make the radius really big so all nodes are connected (just for testing rn. can reduce to 1.1 layer)
        print("senders and receivers:")
        print(senders, receivers)

        # print(l)
        # print(l.shape)
        # print(l[None].shape)
        # print(l[None][None].shape)

        graphs += [
            jraph.GraphsTuple(
                nodes=pos.reshape((4, 3)),  # [num_nodes, 3]
                edges=None,
                globals=label[None],  # [num_graphs]
                senders=senders,  # [num_edges]
                receivers=receivers,  # [num_edges]
                n_node=jnp.array([len(pos)]),  # [num_graphs]
                n_edge=jnp.array([len(senders)]),  # [num_graphs]
            )
        ]

    return jraph.batch(graphs)


class e3jLayer(flax.linen.Module):
    max_l: int
    num_channels: int
    # raw_target_irreps: str
    denominator: float
    # sh_lmax: int = 3

    @flax.linen.compact
    def __call__(self, graphs, positions: Float[Array, 'num_nodes 3'], **kwargs):
        # target_irreps = Irreps(self.raw_target_irreps)

        # this function is called ONCE. you have to update ALL edge features here
        def update_edge_fn(_edge_features, sender_features: jnp.ndarray, receiver_features: jnp.ndarray, _globals):
            # sender features is of shape: [num_edges_communicating, parity_dim, max_l**2, num_channels]

            print("call update_edge_fn")
            # TODO: tensor product with sh???
            # return sender_features
            # the only feature we care in the tetris example is the relative position of the receiver to the sender

            features = positions[graphs.receivers] - positions[graphs.senders]
            # print("sender_features")
            # print(sender_features.shape)
            # the shape of sender features is: (all of the neighbors, 1 ,3)

            # this only maps a 3D vector to a spherical harmonic but what about higher dimensional inputs?
            sh = map_3d_feats_to_spherical_harmonics_repr(
                features,
                normalize=True,
            )
            # TODO: make this more efficient.
            # we want to do a 1 feature to 1 feature tensor product for each edge
            # the result of the tensor product is a tensor of shape: [num_edges_communicating, max_l**2, num_channels]

            # tp = jnp.empty_like(sender_features)
            output_features_shape = list(sender_features.shape)
            output_features_shape[2] = (self.max_l+1)**2
            tp = jnp.empty(output_features_shape)

            for node_idx in range(len(graphs.nodes)):
                node_feats = sender_features[node_idx,:,:,:]
                sh_feats_for_node = sh.slice_ith_feature(node_idx)
                res = tensor_product_v1(Irrep(node_feats), Irrep(sh_feats_for_node), max_l3=self.max_l)
                tp = tp.at[node_idx].set(res)
            # concatenate these arrays along the channel axis (last one)
            # messages = jnp.concatenate([sender_features, tp], axis=-1)
            print("update edge fn finished")
            return tp

        def update_node_fn(node_features, _outgoing_edge_features, incoming_edge_features, _globals):
            # summed_incoming = jnp.sum(incoming_edge_features, axis=0) # no need to do this. jraph's aggregation function by default sums the incoming edge features

            # node_feats = receiver_features / self.denominator
            node_feats = flax.linen.Dense(features=incoming_edge_features.shape[-1], name="linear")(incoming_edge_features)
            # NOTE: removed scalar activation and extra linear layer for now
            return node_feats

        return jraph.GraphNetwork(update_edge_fn, update_node_fn)(graphs)
    

class e3jFinalLayer(flax.linen.Module):
    @flax.linen.compact
    def __call__(self, graphs, **kwargs):

        def update_global_fn(node_features: jnp.ndarray, _edge_features, _globals):
            reshaped_feats = node_features.reshape(node_features.shape[0], -1) # [num_graphs, all_features]
            node_feats = flax.linen.Dense(features=num_classes, name="linear")(reshaped_feats)
            return node_feats

        return jraph.GraphNetwork(update_edge_fn=None, update_node_fn=None, update_global_fn=update_global_fn)(graphs)


class Model(flax.linen.Module):
    @flax.linen.compact
    def __call__(self, graphs):
        # positions = e3nn.IrrepsArray("1o", graphs.nodes)
        positions = graphs.nodes
        graphs = graphs._replace(nodes=jnp.ones((positions.shape[0], 2, 4, 1))) # for each node, ensure it has an empty feature. 3rd dimension is 4 since it's for l=1

        # layers = 2 * ["32x0e + 32x0o + 8x1o + 8x1e + 8x2e + 8x2o"] + ["0o + 7x0e"]

        # for irreps in layers:
        graphs = e3jLayer(max_l=1, num_channels=8, denominator=1)(graphs, positions)
        graphs = e3jLayer(max_l=1, num_channels=8, denominator=1)(graphs, positions)
        graphs = e3jFinalLayer()(graphs)
        logits = graphs.globals

        assert logits.shape == (len(graphs.n_node), num_classes)  # [num_graphs, num_classes]

        return logits


def train(steps=200):
    model = Model()

    # Optimizer
    opt = optax.adam(learning_rate=0.01)

    def loss_fn(params, graphs):
        logits = model.apply(params, graphs)
        labels = graphs.globals  # [num_graphs]

        loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
        loss = jnp.mean(loss)
        return loss, logits

    @jax.jit
    def update_fn(params, opt_state, graphs):
        grad_fn = jax.grad(loss_fn, has_aux=True)
        grads, logits = grad_fn(params, graphs)
        labels = graphs.globals
        accuracy = jnp.mean(jnp.argmax(logits, axis=1) == labels)

        updates, opt_state = opt.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, accuracy

    # dataset
    graphs = tetris()

    # initialize
    init = jax.jit(model.init)
    params = init(jax.random.PRNGKey(0), graphs)
    test_equivariance(model, params)
    opt_state = opt.init(params)

    # compile jit
    wall = time.perf_counter()
    print("compiling...", flush=True)
    _, _, accuracy = update_fn(params, opt_state, graphs)
    print(f"initial accuracy = {100 * accuracy:.0f}%", flush=True)
    print(f"compilation took {time.perf_counter() - wall:.1f}s")

    # train
    wall = time.perf_counter()
    print("training...", flush=True)
    for ith_step in range(steps):
        # if ith_step == 5:
        #     test_equivariance(model, params)
        params, opt_state, accuracy = update_fn(params, opt_state, graphs)

        if accuracy == 1.0:
           break

    print(f"final accuracy = {100 * accuracy:.0f}%")

    # serialize for run_tetris.py
    with open("tetris.mp", "wb") as f:
        f.write(flax.serialization.to_bytes(params))
    
    # serialize for tetris.c
    with open("tetris.bin", "wb") as f:
        for layer in range(3):
            for weights in ["linear", "shortcut"]:
                weight = params["params"][f"Layer_{layer}"][weights]
                weight = np.concatenate([w.ravel() for w in weight.values()])
                f.write(struct.pack(f"{len(weight)}f", *weight))


def test_equivariance(model: Model, params: jnp.ndarray):
    pos = [[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 1, 0]]  # L
    pos = jnp.array(pos, dtype=default_dtype)
    senders, receivers = radius_graph(pos, 10) # make the radius really big so all nodes are connected (just for testing rn. can reduce to 1.1 layer)

    graphs = jraph.batch([
        jraph.GraphsTuple(
            nodes=pos,
            edges=None,
            globals=None,
            senders=senders,  # [num_edges]
            receivers=receivers,  # [num_edges]
            n_node=jnp.array([len(pos)]),  # [num_graphs]
            n_edge=jnp.array([len(senders)]),  # [num_graphs]
        )
    ])

    logits = model.apply(params, graphs)

    rotation_matrix = jnp.array([
        [0.7071, 0.5, 0.5],
        [0, 0.7071, -0.7071],
        [-0.7071, 0.5, 0.5],
    ])
    pos_rotated = jnp.dot(pos, rotation_matrix.T)

    graphs = jraph.batch([
        jraph.GraphsTuple(
            nodes=pos_rotated,
            edges=None,
            globals=None,
            senders=senders,  # [num_edges]
            receivers=receivers,  # [num_edges]
            n_node=jnp.array([len(pos)]),  # [num_graphs]
            n_edge=jnp.array([len(senders)]),  # [num_graphs]
        )
    ])
    # we don't need to rotate the logits since this is a scalar output. it's not a vector
    rotated_logits = model.apply(params, graphs)


    print("logits", logits)
    print("rotated logits", rotated_logits)
    assert jnp.allclose(logits, rotated_logits, atol=1e-2), "model is not equivariant"


if __name__ == "__main__":
    # TODO:(curtis): enable this after
    with jax.disable_jit():
        train()