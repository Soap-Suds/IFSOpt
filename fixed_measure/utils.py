"""
Utility functions for visualization and validation.
"""

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from typing import Union


def visualize_measure(mu: Union[jnp.ndarray, np.ndarray],
                     title: str = "Fixed Measure",
                     figsize: tuple = (10, 10),
                     cmap: str = 'gray_r',
                     save_path: str = None) -> None:
    """
    Visualize a measure as a heatmap.

    Args:
        mu: Measure to visualize (d, d)
        title: Plot title
        figsize: Figure size
        cmap: Colormap name
        save_path: If provided, save figure to this path instead of showing
    """
    mu_np = np.array(mu)

    plt.figure(figsize=figsize)
    plt.imshow(mu_np, origin='lower', extent=[0, 1, 0, 1], cmap=cmap)
    plt.colorbar(label='Measure Density')
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved to {save_path}")
    else:
        plt.show()


def validate_transformation_matrix(M: jnp.ndarray, check_contraction: bool = True) -> bool:
    """
    Validate a 3x3 affine transformation matrix.

    Checks:
    1. Affine on R^2: last row is [0, 0, 1]
    2. Contraction: largest singular value < 1 (optional)
    3. Invertibility: det(M) != 0
    4. Fixed point in [0, 1]^2

    Args:
        M: 3x3 transformation matrix
        check_contraction: Whether to enforce contraction property

    Returns:
        True if valid, False otherwise
    """
    if M.shape != (3, 3):
        return False

    # 1. Affine check
    if not jnp.allclose(M[2, :], jnp.array([0.0, 0.0, 1.0])):
        return False

    # 2. Contraction check
    if check_contraction:
        A = M[:2, :2]
        singular_values = jnp.linalg.svd(A, compute_uv=False)
        if jnp.max(singular_values) >= 1.0:
            return False

    # 3. Invertibility
    if jnp.abs(jnp.linalg.det(M)) < 1e-10:
        return False

    # 4. Fixed point in [0, 1]^2
    A = M[:2, :2]
    b = M[:2, 2]
    I = jnp.eye(2)

    try:
        fixed_point = jnp.linalg.solve(A - I, -b)
        x, y = fixed_point
        if not (0 <= x <= 1 and 0 <= y <= 1):
            return False
    except:
        return False

    return True


def create_sierpinski_ifs() -> tuple:
    """
    Create the standard Sierpinski triangle IFS.

    Returns:
        (F, p): List of 3 transformation matrices and uniform probability vector
    """
    F = [
        jnp.array([
            [0.5, 0.0, 0.0],
            [0.0, 0.5, 0.0],
            [0.0, 0.0, 1.0]
        ], dtype=jnp.float32),
        jnp.array([
            [0.5, 0.0, 0.5],
            [0.0, 0.5, 0.0],
            [0.0, 0.0, 1.0]
        ], dtype=jnp.float32),
        jnp.array([
            [0.5, 0.0, 0.0],
            [0.0, 0.5, 0.5],
            [0.0, 0.0, 1.0]
        ], dtype=jnp.float32)
    ]
    p = jnp.array([1/3, 1/3, 1/3], dtype=jnp.float32)
    return F, p


def create_barnsley_fern_ifs() -> tuple:
    """
    Create the Barnsley fern IFS.

    Returns:
        (F, p): List of 4 transformation matrices and probability vector
    """
    F = [
        # Stem
        jnp.array([
            [0.0, 0.0, 0.0],
            [0.0, 0.16, 0.0],
            [0.0, 0.0, 1.0]
        ], dtype=jnp.float32),
        # Successively smaller leaflets
        jnp.array([
            [0.85, 0.04, 0.0],
            [-0.04, 0.85, 0.16],
            [0.0, 0.0, 1.0]
        ], dtype=jnp.float32),
        # Largest left-hand leaflet
        jnp.array([
            [0.2, -0.26, 0.0],
            [0.23, 0.22, 0.16],
            [0.0, 0.0, 1.0]
        ], dtype=jnp.float32),
        # Largest right-hand leaflet
        jnp.array([
            [-0.15, 0.28, 0.0],
            [0.26, 0.24, 0.044],
            [0.0, 0.0, 1.0]
        ], dtype=jnp.float32)
    ]
    p = jnp.array([0.01, 0.85, 0.07, 0.07], dtype=jnp.float32)
    return F, p
