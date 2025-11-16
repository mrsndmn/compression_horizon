import argparse

import matplotlib.pyplot as plt
import torch

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--params_path", type=str, required=True)
    args = parser.parse_args()

    params_path = args.params_path
    params = torch.load(params_path)
    print(params.keys())

    # {
    #     "control_point": c_param.cpu(),
    #     "num_compression_tokens": int(e0.shape[0]),
    #     "hidden_size": int(e0.shape[1]),
    #     "endpoints": {
    #         "e0": e0.detach().cpu(),
    #         "e1": e1.detach().cpu(),
    #     },
    #     "model_checkpoint": model_name,
    # },

    e0 = params["endpoints"]["e0"]
    e1 = params["endpoints"]["e1"]
    c = params["control_point"]
    print("e0", e0.shape, e0.min(), e0.max(), e0.norm())
    print("e1", e1.shape, e1.min(), e1.max(), e1.norm())
    print("c", c.shape, c.min(), c.max(), c.norm())

    linear_interpolation = (e0 + e1) / 2

    # num_compression_tokens = params["num_compression_tokens"]
    # hidden_size = params["hidden_size"]
    # model_checkpoint = params["model_checkpoint"]

    # Print pairwise distances between e0, e1, c, and linear_interpolation
    all_embeddings = torch.stack([e0, linear_interpolation, e1, c]).squeeze(1)
    distances = torch.pairwise_distance(all_embeddings, all_embeddings, p=2)
    print(distances)

    c_normalized = (c - e0) / (e1 - e0)

    # plot values distribution for c_normalized
    plt.figure(figsize=(7, 4))
    plt.hist(c[:, :].squeeze().detach().cpu().numpy(), bins=100, alpha=0.5, label="c_normalized")
    plt.hist(e0[:, :].squeeze().detach().cpu().numpy(), bins=100, alpha=0.5, label="e0")
    plt.hist(e1[:, :].squeeze().detach().cpu().numpy(), bins=100, alpha=0.5, label="e1")
    plt.xlabel("value")
    plt.ylabel("count")
    plt.title("Values distribution for c_normalized")
    plt.legend()
    plt.tight_layout()
    plt.savefig("artifacts/visualizations/bezier_values_distribution.png", dpi=150)
    plt.close()
    print("Saved plot to: artifacts/visualizations/bezier_values_distribution.png")
