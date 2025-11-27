import torch
import copy
import numpy as np
import torch.nn as nn


def get_weights(model):
    """Return detached clones so downstream sweeps cannot mutate in-place state."""
    return [p.detach().clone() for p in model.parameters()]


def set_weights(model, weights):
    """Safely copy precomputed tensors back into model parameters."""
    with torch.no_grad():
        for p, w in zip(model.parameters(), weights):
            p.copy_(w)

def normalize_direction(direction, weights, norm='filter'):
    if norm == 'filter':
        # Filter-wise normalization
        for d, w in zip(direction, weights):
            if d.dim() > 1:
                # Normalize each filter (output channel) independently
                # d shape: (out_channels, in_channels, k, k)
                # We want norm over (in_channels, k, k)
                d_flat = d.view(d.size(0), -1)
                w_flat = w.view(w.size(0), -1)
                
                d_norm = d_flat.norm(dim=1, keepdim=True).view(-1, *([1]*(d.dim()-1)))
                w_norm = w_flat.norm(dim=1, keepdim=True).view(-1, *([1]*(w.dim()-1)))
                
                d.div_(d_norm + 1e-10)
                d.mul_(w_norm)
            else:
                d.div_(d.norm() + 1e-10)
                d.mul_(w.norm())
    elif norm == 'layer':
        # Layer-wise normalization
        for d, w in zip(direction, weights):
            d.div_(d.norm())
            d.mul_(w.norm())
    return direction

def create_random_direction(model):
    weights = get_weights(model)
    direction = [torch.randn_like(w) for w in weights]
    return direction

def get_1d_interpolation(model, loader, start_weights, end_weights, steps=20, device='cuda'):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    
    alphas = np.linspace(-1, 2, steps)
    losses = []
    accs = []
    
    direction = [e - s for s, e in zip(start_weights, end_weights)]
    
    for alpha in alphas:
        current_weights = [s + alpha * d for s, d in zip(start_weights, direction)]
        set_weights(model, current_weights)
        
        loss, acc = evaluate(model, loader, criterion, device)
        losses.append(loss)
        accs.append(acc)

    # restore trained parameters before returning
    set_weights(model, start_weights)

    return alphas, losses, accs

def get_2d_contour(model, loader, center_weights, x_dir, y_dir, steps=20, device='cuda'):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    
    x = np.linspace(-1, 1, steps)
    y = np.linspace(-1, 1, steps)
    X, Y = np.meshgrid(x, y)
    
    losses = np.zeros_like(X)
    
    for i in range(steps):
        for j in range(steps):
            delta = [X[i, j] * xd + Y[i, j] * yd for xd, yd in zip(x_dir, y_dir)]
            current_weights = [c + d for c, d in zip(center_weights, delta)]
            set_weights(model, current_weights)

            loss, _ = evaluate(model, loader, criterion, device)
            losses[i, j] = loss

    # reset back to contour center
    set_weights(model, center_weights)

    return X, Y, losses

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return total_loss / total, 100. * correct / total
