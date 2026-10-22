# Optimization for ML

Homework for the *Optimization for Machine Learning* course (Université Paris Cité, 2025), implemented in PyTorch.

## Contents

**1. Nesterov Accelerated Gradient: Speed Restart**
- Implementation of NAG with speed restart (Su, Boyd & Candès, 2016) and NAG with vanishing friction (b = 3, no restart)
- Comparison on the 2D Rosenbrock function (b = 100), with step size α = 1/L
- Speed restart converges faster and without the oscillations of vanishing friction

**2. A scaling law for the step size of gradient descent**
- One-hidden-layer ReLU network fitted to noisy data y = sin(t) + ε
- Grid search for the optimal step size α* under a fixed iteration budget (T = 3000)
- Fit of a power law α*(d) ≈ ν · d^γ as the hidden width d grows

## References
- W. Su, S. Boyd, E. Candès, *A Differential Equation for Modeling Nesterov's Accelerated Gradient Method*, JMLR 2016
- B. O'Donoghue, E. Candès, *Adaptive Restart for Accelerated Gradient Schemes*, FoCM 2015
