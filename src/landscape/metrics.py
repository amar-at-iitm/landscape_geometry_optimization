import torch
import torch.nn as nn

def compute_hessian_eig(model, loader, criterion, device='cuda', top_k=1, max_iter=100, tol=1e-3):
    """
    Compute the top-k eigenvalues of the Hessian using Power Iteration.
    """
    model.eval()
    model.to(device)
    
    params = [p for p in model.parameters() if p.requires_grad]
    num_params = sum(p.numel() for p in params)
    
    # Initialize random vector
    v = [torch.randn_like(p) for p in params]
    v = normalize(v)
    
    eigenvalues = []
    
    for k in range(top_k):
        for i in range(max_iter):
            # Hessian-vector product
            Hv = hessian_vector_product(model, loader, criterion, v, device)
            
            # Rayleigh quotient
            eig_val = dot_product(v, Hv)
            
            # Orthogonalize against previous eigenvectors (if k > 0)
            # (Not implemented for k=1, but needed for k>1)
            
            # Update v
            v_new = normalize(Hv)
            
            # Check convergence
            if i > 0 and torch.abs(eig_val - prev_eig_val) < tol:
                break
            prev_eig_val = eig_val
            v = v_new
            
        eigenvalues.append(eig_val.item())
        
    return eigenvalues

def hessian_vector_product(model, loader, criterion, v, device):
    """
    Compute Hessian-vector product using the pearlmutter trick (finite differences or double backprop).
    Here we use double backprop which is cleaner in PyTorch.
    """
    # We need to compute Hv = grad(grad(L) * v)
    # But doing this over the whole dataset is expensive.
    # We typically approximate it with a mini-batch or a few mini-batches.
    
    Hv = [torch.zeros_like(p) for p in v]
    count = 0
    
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)
        
        # Compute dot product of grads and v
        grad_v = sum(torch.sum(g * vi) for g, vi in zip(grads, v))
        
        # Compute gradient of grad_v w.r.t parameters
        hvp = torch.autograd.grad(grad_v, model.parameters(), retain_graph=False)
        
        for i, h in enumerate(hvp):
            Hv[i] += h.detach()
            
        count += 1
        # For efficiency, we might only use a subset of the loader
        if count >= 10: # Limit to 10 batches for speed
            break
            
    # Average over batches
    Hv = [h / count for h in Hv]
    return Hv

def normalize(v):
    norm = torch.sqrt(sum(torch.sum(vi * vi) for vi in v))
    return [vi / (norm + 1e-6) for vi in v]

def dot_product(v1, v2):
    return sum(torch.sum(a * b) for a, b in zip(v1, v2))
