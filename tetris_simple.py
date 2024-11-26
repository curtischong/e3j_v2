# this is the same as tetris.py, except it doesn't use message passing and only uses basis functions to test equivariance
# the model is smaller and faster to train


from irrep import Irreps
from model import LinearLayer
from spherical_harmonics import map_3d_feats_to_spherical_harmonics_repr
import tetris
import torch.nn as nn
import torch


class TensorDense(nn.Module):
    # perform a tensor product with two linear projections of itself
    def __init__(self, in_irreps_id: str, out_irreps_id: str):
        super().__init__()
        self.linear1 = LinearLayer(in_irreps_id, in_irreps_id)
        self.linear2 = LinearLayer(in_irreps_id, in_irreps_id)

        tensor_product_output_irreps_id = Irreps.get_tensor_product_output_irreps_id(
            in_irreps_id, in_irreps_id
        )

        self.linear3 = LinearLayer(
            tensor_product_output_irreps_id, out_irreps_id
        )  # this last one is to combine the result of the tensor product into the target irreps i

    def forward(self, x: Irreps) -> Irreps:
        x1: Irreps = self.linear1(x)
        x2: Irreps = self.linear2(x)
        tp = x1.tensor_product(x2)
        return self.linear3(tp)


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.tensor_dense = TensorDense(3)
        self.output_mlp = nn.Linear(7)

    def forward(self, positions):
        x = basis_functions(positions)
        x = self.tensor_dense(positions)
        return self.output_mlp(x)


def train_tetris_simple() -> None:
    x, y = tetris()
    model = SimpleModel()

    for step in range(300):
        for i, positions in enumerate(x):
            pred = model(positions)
            loss = (pred - y[i]).pow(2).sum()


if __name__ == "__main__":
    train_tetris_simple()
