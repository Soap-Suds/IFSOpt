"""
IFS Fixed Measure Solver
========================

High-performance JAX implementation for computing fixed measures of
Iterated Function Systems (IFS) using the Banach fixed-point theorem.

Optimized for use in optimization loops with minimal recompilation overhead.
"""

from ifs_solver.core import FixedMeasureSolver

__all__ = ['FixedMeasureSolver']
__version__ = '1.0.0'
