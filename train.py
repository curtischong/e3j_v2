# Adapted from https://github.com/e3nn/e3nn-jax/blob/main/examples/tetris_point.py
#  * removed scalar non-linearity for now

import time

import flax
import flax.serialization
import jax
import jax.numpy as jnp
import jraph
import optax

from constants import default_dtype
from graph_utils import radius_graph
from irrep import Irrep
from spherical_harmonics import map_3d_feats_to_spherical_harmonics_repr
from tensor_product import tensor_product_v1, tensor_product_v2
from jaxtyping import Array, Float
from utils import plot_3d_coords


shapes = [
    # [[0, 0, 0], [0, 0, 1], [1, 0, 0], [1, 1, 0]],  # chiral_shape_1 # curtis: chiral_shape_1 and chiral_shape_2 are the same except I think chiral_shape_2 is reflected. Either way, it makes the output irreps harder to predict (need a o1 output to differentiate between the two tetrises.
    # Since I'm lazy, and want this to be as simple as possible, I will just have one chiral shape
    [[1, 1, 1], [1, 1, 2], [2, 1, 1], [2, 0, 1]],  # chiral_shape_2
    # [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]],  # square
    # [[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 0, 3]],  # line
    # [[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]],  # corner
    [[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 1, 0]],  # L
    # [[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 1, 1]],  # T # curtis: how is this a T???? doens't it need 5 points?
    # [[0, 0, 0], [1, 0, 0], [1, 1, 0], [2, 1, 0]],  # zigzag
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
    denominator: float

    def setup(self):
        # Define any layers or parameters here
        self.linear = flax.linen.Dense(features=1, name="linear", use_bias=False)

    def __call__(self, graphs, positions: Float[Array, 'num_nodes 3'], **kwargs):
        def update_edge_fn(_edge_features, sender_features: jnp.ndarray, receiver_features: jnp.ndarray, _globals):
            # Sender features are of shape: [num_edges_communicating, parity_dim, max_l**2, num_channels]
            message_direction_feature = positions[graphs.receivers] - positions[graphs.senders]

            # Map 3D vector to spherical harmonics representation
            sh = map_3d_feats_to_spherical_harmonics_repr(
                message_direction_feature,
                normalize=True,
            )

            output_features_shape = list(sender_features.shape)
            output_features_shape[2] = (self.max_l + 1)**2
            tp = jnp.empty(output_features_shape)

            # Perform tensor product for each edge
            for node_idx in range(len(graphs.nodes)):
                node_feats = sender_features[node_idx, :, :, :]
                sh_feats_for_node = Irrep.slice_ith_feature(sh, node_idx)
                # res = tensor_product_v1(node_feats, sh_feats_for_node, max_l3=self.max_l)
                res = tensor_product_v2(node_feats, sh_feats_for_node)
                tp = tp.at[node_idx].set(res)
            return tp

        def update_node_fn(old_node_features, _outgoing_edge_features, incoming_edge_features, _globals):
            # Perform node feature update
            node_features = incoming_edge_features / self.denominator
            node_features = self.linear(node_features)
            # print("incoming edge features", incoming_edge_features)
            return node_features

        return jraph.GraphNetwork(update_edge_fn, update_node_fn)(graphs)

    

class e3jFinalLayer(flax.linen.Module):
    # do NOT use @flax.linen.compact because otherwise, the linear layer will reinitialize and have diff weights
    # I think it's because we're defining it inside the update_global_fn function, not directly inside the __call__ function???
    # or maybe it's just a bug when we're not JITing the model (which I do during testing)
    def setup(self):
        # Define the linear layer here to avoid compact-style
        self.linear = flax.linen.Dense(features=num_classes, name="linear")

    def __call__(self, graphs, **kwargs):
        def update_global_fn(node_features: jnp.ndarray, _edge_features, _globals):
            # Extract only the even scalar features
            scalar_feats = node_features[:, 0, 0, :]  # [num_graphs, 1, 1, num_channels]
            scalar_feats = scalar_feats.reshape(scalar_feats.shape[0], -1)  # [num_graphs, all_features]
            res = self.linear(scalar_feats)
            return res

        return jraph.GraphNetwork(update_edge_fn=None, update_node_fn=None, update_global_fn=update_global_fn)(graphs)

class Model(flax.linen.Module):
    @flax.linen.compact
    def __call__(self, graphs):
        positions = graphs.nodes
        graphs = graphs._replace(nodes=jnp.ones((positions.shape[0], 2, 4, 1))) # for each node, ensure it has an empty feature. 3rd dimension is 4 since it's for l=1

        # for irreps in layers:
        graphs = e3jLayer(max_l=3, denominator=1)(graphs, positions)
        # TODO: put a nonlinearity here. otherwise it's just as good as putting a single linear layer
        graphs = e3jLayer(max_l=5, denominator=1)(graphs, positions)
        graphs = e3jFinalLayer()(graphs)
        logits = graphs.globals

        assert logits.shape == (len(graphs.n_node), num_classes), f"logits shape: {logits.shape}, num_nodes: {len(graphs.n_node)}, num_classes: {num_classes}"

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
        # print(f"accuracy = {100 * accuracy:.0f}%, logits = {logits}")
        return params, opt_state, accuracy, logits

    # dataset
    graphs = tetris()

    # initialize
    init = jax.jit(model.init)
    params = init(jax.random.PRNGKey(0), graphs)
    # test_equivariance(model, params)
    opt_state = opt.init(params)

    # compile jit
    wall = time.perf_counter()
    print("compiling...", flush=True)
    _, _, accuracy, _ = update_fn(params, opt_state, graphs)
    print(f"initial accuracy = {100 * accuracy:.0f}%", flush=True)
    print(f"compilation took {time.perf_counter() - wall:.1f}s")

    # train
    wall = time.perf_counter()
    print("training...", flush=True)
    for _ith_step in range(steps):
        params, opt_state, accuracy, logits = update_fn(params, opt_state, graphs)
        print(f"step {_ith_step}, accuracy = {100 * accuracy:.0f}%, logits = {logits}")

        if accuracy == 1.0:
           break

    print(f"final accuracy = {100 * accuracy:.0f}%")

    # serialize for run_tetris.py
    with open("tetris.mp", "wb") as f:
        f.write(flax.serialization.to_bytes(params))
    
def prepare_single_graph(pos: jnp.ndarray, radius: float) -> jraph.GraphsTuple:
    senders, receivers = radius_graph(pos, radius) # make the radius really big so all nodes are connected (just for testing rn. can reduce to 1.1 layer)

    return jraph.batch([
        jraph.GraphsTuple(
            nodes=pos,
            edges=None,
            globals=None,
            senders=senders,
            receivers=receivers,
            n_node=jnp.array([len(pos)]),
            n_edge=jnp.array([len(senders)]),
        )
    ])

def get_rotation_matrix(roll: float, pitch: float, yaw: float) -> jnp.ndarray:
    """
    Returns a 3x3 rotation matrix given roll, pitch, and yaw angles.
    
    Parameters:
    - roll: Rotation angle around the x-axis in radians.
    - pitch: Rotation angle around the y-axis in radians.
    - yaw: Rotation angle around the z-axis in radians.
    
    Returns:
    - A 3x3 JAX array representing the rotation matrix.
    """
    # Rotation matrix around the x-axis
    Rx = jnp.array([
        [1, 0, 0],
        [0, jnp.cos(roll), -jnp.sin(roll)],
        [0, jnp.sin(roll), jnp.cos(roll)]
    ])
    
    # Rotation matrix around the y-axis
    Ry = jnp.array([
        [jnp.cos(pitch), 0, jnp.sin(pitch)],
        [0, 1, 0],
        [-jnp.sin(pitch), 0, jnp.cos(pitch)]
    ])
    
    # Rotation matrix around the z-axis
    Rz = jnp.array([
        [jnp.cos(yaw), -jnp.sin(yaw), 0],
        [jnp.sin(yaw), jnp.cos(yaw), 0],
        [0, 0, 1]
    ])
    
    # Combine the rotations: R = Rz * Ry * Rx
    R = Rz @ Ry @ Rx
    return R


def test_equivariance(model: Model, params: jnp.ndarray):
    pos = [[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 1, 0]]  # L
    pos = jnp.array(pos, dtype=default_dtype)

    graphs = prepare_single_graph(pos, 11)

    logits = model.apply(params, graphs)

    max_distance = 0
    for angle1 in jnp.arange(0, 1, 0.2):
        for angle2 in jnp.arange(1, 2, 0.2):
            for angle3 in jnp.arange(0, 1, 0.2):
                rotation_matrix = get_rotation_matrix(jnp.pi*angle1, jnp.pi*angle2, jnp.pi*angle3)
                # plot_3d_coords(pos)
                pos_rotated = jnp.dot(pos, rotation_matrix.T) # we transpose and matrix multiply from the left side because python's vectors are row vectors, NOT column vectors. so we can't just do y=Ax
                # plot_3d_coords(pos_rotated)

                graphs = prepare_single_graph(pos_rotated, 11)

                # we don't need to rotate the logits since this is a scalar output. it's not a vector
                rotated_logits = model.apply(params, graphs)
                print("rotated logits", rotated_logits)

                rotational_equivariance_error = jnp.mean(jnp.abs(logits - rotated_logits))
                print("logit diff distance", round(rotational_equivariance_error,7), "\tangle1", round(angle1,6), "\tangle2", round(angle2,6), "\tangle3", round(angle3,6))
                max_distance = max(max_distance, rotational_equivariance_error)
    print("max distance", max_distance)
    assert jnp.allclose(logits, rotated_logits, atol=1e-2), "model is not equivariant"
    print("the model is equivariant!")


if __name__ == "__main__":
    # TODO:(curtis): enable this after
    # with jax.disable_jit():
    train()