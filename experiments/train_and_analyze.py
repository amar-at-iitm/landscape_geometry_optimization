import argparse
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from src.models.models import MLP, ResNet20
from src.training.trainer import train
from src.landscape.visualization import (
    create_random_direction,
    get_1d_interpolation,
    get_2d_contour,
    get_weights,
    normalize_direction,
)
from src.landscape.metrics import compute_hessian_eig

from data_preparation import prepare_dataloaders, build_analysis_loader

DEFAULT_SAVE_DIR = os.path.join(os.path.dirname(__file__), 'outputs')


def main():
    parser = argparse.ArgumentParser(description='Training + Landscape Analysis')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--model', type=str, default='mlp', choices=['mlp', 'resnet20'])
    parser.add_argument('--dataset', type=str, default='synthetic', choices=['synthetic', 'cifar10'])
    parser.add_argument('--data_root', type=str)
    parser.add_argument('--download_data', action='store_true')
    parser.add_argument('--save_dir', type=str, default=DEFAULT_SAVE_DIR)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--analysis_samples', type=int, default=512)
    parser.add_argument('--hessian_iters', type=int, default=20)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    print("==> Preparing data (no modifications)...")
    trainloader, testloader, trainset = prepare_dataloaders(args)

    print("==> Building model..")
    net = MLP() if args.model == 'mlp' else ResNet20()

    print("==> Training..")
    checkpoint_path = os.path.join(args.save_dir, 'model.pth')
    train(net, trainloader, testloader, epochs=args.epochs, lr=args.lr, save_path=checkpoint_path, device=device)

    if not os.path.exists(checkpoint_path):
        torch.save(net.state_dict(), checkpoint_path)
    net.load_state_dict(torch.load(checkpoint_path, map_location=device))

    print("==> Landscape Analysis..")
    analysis_loader = build_analysis_loader(trainset, args)
    criterion = nn.CrossEntropyLoss()

    print("Computing Hessian Eigenvalue...")
    eig_vals = compute_hessian_eig(
        net,
        analysis_loader,
        criterion,
        device=device,
        top_k=1,
        max_iter=args.hessian_iters,
    )
    print(f"Top Hessian Eigenvalue: {eig_vals[0]:.4f}")
    plot_hessian(eig_vals, os.path.join(args.save_dir, 'hessian.png'))

    print("Generating 1D interpolation...")
    start_weights = get_weights(net)
    direction = normalize_direction(create_random_direction(net), start_weights, norm='filter')
    end_weights = [w + d for w, d in zip(start_weights, direction)]
    alphas, losses, _ = get_1d_interpolation(
        net,
        analysis_loader,
        start_weights,
        end_weights,
        steps=20,
        device=device,
    )
    plot_1d_curve(alphas, losses, os.path.join(args.save_dir, 'interp_1d.png'))

    print("Generating 2D contour...")
    x_dir = normalize_direction(create_random_direction(net), start_weights, norm='filter')
    y_dir = normalize_direction(create_random_direction(net), start_weights, norm='filter')
    X, Y, Z = get_2d_contour(
        net,
        analysis_loader,
        start_weights,
        x_dir,
        y_dir,
        steps=10,
        device=device,
    )
    plot_2d_contour(X, Y, Z, os.path.join(args.save_dir, 'contour_2d.png'))

    print("Analysis Complete.")


def plot_hessian(eig_vals, path):
    plt.figure()
    plt.bar([r'$\lambda_1$'], eig_vals)
    plt.ylabel('Value')
    plt.title('Top Hessian Eigenvalue')
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_1d_curve(alphas, losses, path):
    plt.figure()
    plt.plot(alphas, losses)
    plt.title('1D Loss Landscape')
    plt.xlabel('Alpha')
    plt.ylabel('Loss')
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_2d_contour(X, Y, Z, path):
    plt.figure()
    contour = plt.contourf(X, Y, Z, levels=20)
    plt.colorbar(contour)
    plt.title('2D Loss Landscape')
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


if __name__ == '__main__':
    main()
