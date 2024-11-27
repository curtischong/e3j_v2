"""Classify tetris using gate activation function

Implement a equivariant model using gates to fit the tetris dataset
Exact equivariance to :math:`E(3)`

>>> test()
"""

import os
import torch

from tetris_data import tetris
from o3.model import ActivationLayer, Layer
from utils.geometric_utils import avg_irreps_with_same_id
from utils.model_utils import seed_everything


from utils.dummy_data_utils import create_irreps_with_dummy_data
from utils.graph_utils import to_graph
from utils.constants import default_dtype


class Model(torch.nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.starting_irreps_id = "1x0e"  # each node starts with a dummy 1x0e irrep
        self.radius = 11

        # first layer
        self.layer1 = Layer(self.starting_irreps_id, "5x0e + 5x1o")
        self.activation_layer1 = ActivationLayer("GELU", "5x0e + 5x1o")
        self.layer2 = Layer("5x0e + 5x1o", "10x0e")
        self.activation_layer2 = ActivationLayer("GELU", "10x0e")

        # output layer
        num_scalar_features = 10  # since the output of layer3 is 8x
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

        # now pool the features on each node to generate the final output irreps
        pooled_feats = avg_irreps_with_same_id(x)
        scalar_feats = [irrep.data for irrep in pooled_feats.get_irreps_by_id("0e")]
        x = self.output_mlp(torch.cat(scalar_feats))
        return self.softmax(x)


def main() -> None:
    x, y = tetris()
    # train_x, train_y = x[1:], y[1:]  # dont train on both chiral shapes
    train_x, train_y = x, y

    x, y = tetris()
    test_x, test_y = x, y

    model = Model(num_classes=train_y.shape[1])

    print("Built a model:")
    print(model)
    print(list(model.parameters()))
    print("------------------end of params--------------------")

    optim = torch.optim.Adam(model.parameters(), lr=3e-3)

    for step in range(300):
        optim.zero_grad()
        loss = torch.zeros(1)
        for i, positions in enumerate(train_x):
            pred: torch.Tensor = model(positions)
            loss += (pred - train_y[i]).pow(2).sum()
        loss.backward()
        print(f"loss {loss.item():<10.20f}")

        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optim.step()

        if step % 10 == 0:
            for name, param in model.named_parameters():
                if param.grad is not None:
                    print(f"{name}_______", param.grad.tolist())
                else:
                    print(f"{name}_______")

            current_accuracy = 0
            for i, positions in enumerate(test_x):
                pred = model(positions)
                print("raw pred", pred.tolist())
                one_hot = torch.zeros_like(pred)
                predicted_class = torch.argmax(pred, dim=0)
                one_hot[predicted_class] = 1
                print("pred", one_hot.tolist())
                print("targ", test_y[i].tolist())
                accuracy = (
                    model(positions)
                    .argmax(dim=0)
                    .eq(test_y[i].argmax(dim=0))
                    .double()
                    .item()
                )
                current_accuracy += accuracy
            current_accuracy /= len(test_x)
            print(f"epoch {step:5d} | {100 * current_accuracy:5.1f}% accuracy")
            if current_accuracy == 1.0:
                break

    model_location = "tetris.mp"
    print("saving model to", model_location)
    torch.save(model.state_dict(), model_location)


def profile() -> None:
    data, labels = tetris()
    # data = data.to(device="cuda")
    # labels = labels.to(device="cuda")

    f = Model(labels.shape[1])
    # f.to(device="cuda")

    optim = torch.optim.Adam(f.parameters(), lr=0.05)

    called_num = [0]

    def trace_handler(p) -> None:
        os.makedirs("profiles", exist_ok=True)
        # print(p.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1))
        p.export_chrome_trace("profiles/test_trace_" + str(called_num[0]) + ".json")
        called_num[0] += 1

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            # torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(wait=50, warmup=1, active=1),
        on_trace_ready=trace_handler,
    ) as p:
        for _ in range(52):
            for i, positions in enumerate(data):
                optim.zero_grad()
                pred = f(positions)
                loss = (pred - labels[i]).pow(2).sum()

                loss.backward()
                optim.step()

                p.step()


if __name__ == "__main__":
    seed_everything(143)
    # profile()
    main()
    # equivariance_test()
