# Loss-Landscape Geometry & Optimization Dynamics

This note summarizes the theoretical framing that guided the implementation inside this repository. It links geometric properties of the empirical risk to optimization behavior, generalization, and architectural choices.

## 1. Setup

Let $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^N$ be the training set and $\mathcal{L}(\theta) = \frac{1}{N}\sum_i \ell(f_\theta(x_i), y_i)$ the empirical risk for parameters $\theta$. We focus on smooth losses (cross-entropy) so that the gradient $g = \nabla \mathcal{L}(\theta)$ and Hessian $H = \nabla^2 \mathcal{L}(\theta)$ exist almost everywhere.

Key geometric descriptors:
- **Gradient norm** $\|g\|$ quantifies immediate progress under first-order methods.
- **Hessian spectrum** $\{\lambda_k\}$ captures curvature; large positive eigenvalues indicate sharp directions.
- **Directional sharpness** $\phi(v) = v^\top H v$ for unit vector $v$ measures local quadratic growth along $v$.

## 2. Optimization Dynamics

Stochastic Gradient Descent with learning rate $\eta$ and batch size $B$ updates
$$
\theta_{t+1} = \theta_t - \eta(g_t + \xi_t),
$$
where $\xi_t$ models gradient noise with covariance $\Sigma \approx \frac{1}{B}(H C H)$ for mini-batch covariance $C$. Two consequences follow:

1. **Noise-driven escaping** - In sharp basins (large $\lambda_{max}$) the stochastic term injects enough energy to exit, biasing SGD toward flatter regions where $\lambda_{max}$ is smaller.
2. **Implicit regularization** - Linearized analysis shows stationary distribution variance proportional to $\eta B^{-1} H^{-1}$, so wide valleys (small eigenvalues) retain mass and correspond to solutions SGD visits more often.

The power-iteration Hessian probe implemented here approximates $\lambda_{max}$, providing a diagnostic for how "sharp" the final iterate is. Lower eigenvalues typically correlate with better test accuracy after normalization.

## 3. Generalization Links

Generalization bounds based on PAC-Bayesian flatness or local-Rademacher complexity depend on curvature quantities such as $\text{tr}(H)$ or norms of $H$. Roughly, if perturbations $\delta$ satisfying $\|\delta\|_2 \leq \epsilon$ change the loss by only $O(\epsilon^2 \lambda_{max})$, then weight noise (or SGD noise) does not incur large risk, leading to tighter bounds. Hence, reporting the Hessian spectrum and 1-D/2-D traversals provides empirical evidence of flatness.

## 4. Architectural Effects

Convolutional residual networks generally exhibit smoother landscapes than fully-connected MLPs at the same depth due to:
- Parameter sharing, which constrains the effective dimensionality of perturbations.
- Skip connections, which stabilize gradients by keeping Jacobians closer to identity, reducing extreme eigenvalues of $H$.

The framework allows switching between MLP and ResNet20 to observe these differences experimentally.

## 5. Predicting Optimization Difficulty

We can approximate the expected decrease in loss along direction $v$ after one SGD step via the quadratic model
$$
\Delta \mathcal{L} \approx -\eta v^\top g + \tfrac{1}{2}\eta^2 v^\top H v.
$$
If $v$ aligns with a large eigenvalue, the second term dominates, suggesting smaller steps or adaptive learning rates are necessary. By sampling random directions (as done in the 1-D and 2-D plots) we reveal anisotropy: elongated valleys indicate some directions allow aggressive learning rates, whereas narrow ridges warn of instability.

## 6. Practical Diagnostics Implemented

| Diagnostic | File | Purpose |
| --- | --- | --- |
| Top Hessian eigenvalue (power iteration) | `src/landscape/metrics.py` | Quantifies curvature sharpness and correlates with generalization |
| 1-D interpolation plot | `src/landscape/visualization.py` | Shows loss along a linear path through parameter space, highlighting barriers |
| 2-D contour | `src/landscape/visualization.py` | Reveals basin shape and anisotropy |
| Training dynamics logging | `src/training/trainer.py` | Monitors convergence needed for interpreting geometry |

Together with the theoretical links above, these probes create a rigorous yet computationally feasible framework for reasoning about why certain optimizers and architectures reach generalizable minima despite the non-convex landscape.
