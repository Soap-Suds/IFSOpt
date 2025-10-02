"""
Surrogate Gradient Computation for IFS Optimization
====================================================

High-performance JAX implementation for computing surrogate gradients
for Iterated Function Systems (IFS) optimization using optimal transport.

Optimized for use in optimization loops with minimal recompilation overhead.
"""

from surrogate_gradients.core import SurrogateGradientSolver

__all__ = ['SurrogateGradientSolver']
__version__ = '1.0.0'
