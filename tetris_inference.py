import torch

from model import Model
from tetris import random_rotate_data, tetris


def trained_model_equivariance_test() -> None:
    x, y = tetris()
    model = Model(num_classes=y.shape[1])
    model.load_state_dict(torch.load("tetris.mp"))

    # x, y = x[1:], y[1:]  # predict both chiral shapes

    num_equivariance_tests = 10
    for _step in range(num_equivariance_tests):
        for i, positions in enumerate(x):
            out = model(positions)
            predicted_class = torch.argmax(out, dim=0)
            target_class = y[i].argmax(dim=0)
            assert predicted_class == target_class
            # print("predicted class", predicted_class)

            out2 = model(random_rotate_data(positions))
            print(f"class: {target_class.item()}", (out - out2).abs().sum())
            # predicted_class2 = torch.argmax(out2, dim=0)
            # assert predicted_class2 == y[i].argmax(dim=0)
            # print("out", out)
            # print("out2", out2)
    print("the model is equivariant!")


if __name__ == "__main__":
    trained_model_equivariance_test()
