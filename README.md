# neural-ode
Implementation of Neural ODEs paper ("Neural Ordinary Differential Equations", Chen et al., 2018). For original implementation, see ``torchdiffeq`` package at ``https://github.com/rtqichen/torchdiffeq``.

For a pared down example of how to train an ODE-net on MNIST, see ``mnist_results.ipynb``. Note that for this example, I do not implement the adjoint method, but backpropagate directly through the ODE-net (this corresponds to the ''RK-Net'' in Table I of the Neural ODEs paper). To compare direct backpropagation to the adjoint method, I have also the adjoint method from the ``torchdiffeq`` package.

For a pared down implementation of the adjoint method, see ``adjoint.ipynb``. In this notebook, I show how to compute the gradient for an ODE solver (in this case, Runge-Kutta), and update parameters, without backpropagating through the solver. The gradients from this custom adjoint method are compared to those obtained from direct backpropagation, and to those obtained using the ``torchdiffeq`` package.
