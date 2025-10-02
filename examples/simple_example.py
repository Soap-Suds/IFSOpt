#!/usr/bin/env python3
"""
Simplest possible example - generate a Sierpinski triangle.
"""

import sys
sys.path.insert(0, '/home/spud/IFSOpt')

import jax.numpy as jnp
from fixed_measure import FixedMeasureSolver
from ifs_solver.utils import create_sierpinski_ifs, visualize_measure

# Create solver
solver = FixedMeasureSolver(d=512, eps=1e-6)

# Get Sierpinski IFS
F, p = create_sierpinski_ifs()

# Solve
print("Computing fixed measure...")
mu = solver.solve(F=F, p=p)

# Visualize
visualize_measure(mu, title="Sierpinski Triangle")
