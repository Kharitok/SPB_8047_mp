import numpy as np
import jax.numpy as jnp
from typing import Union


def constrained_geomspace(
    start: float,
    stop: float,
    relative_step: float = 0.1,
    min_step: float = None,
    max_step: float = None,
    use_jax: bool = True,
) -> Union[np.ndarray, jnp.ndarray]:
    """
    Generate a sequence with approximately constant relative step size,
    subject to minimum and maximum step size constraints.

    This function creates a spacing similar to np.geomspace but allows you to
    control both the relative step size and absolute step size limits.

    Parameters
    ----------
    start : float
        Starting value of the sequence (must be > 0)
    stop : float
        End value of the sequence (must be > start)
    relative_step : float, optional
        Target relative step size (fraction). Default is 0.1 (10% increase each step).
        For example, 0.2 means each step is 20% larger than the previous value.
    min_step : float, optional
        Minimum absolute step size. If None, no minimum constraint is applied.
        Useful to prevent very small steps at small values.
    max_step : float, optional
        Maximum absolute step size. If None, no maximum constraint is applied.
        Useful to prevent very large steps at large values.
    use_jax : bool, optional
        Whether to return JAX array (True) or NumPy array (False). Default is True.

    Returns
    -------
    sequence : jnp.ndarray or np.ndarray
        Array of values with constrained relative spacing

    Notes
    -----
    - The algorithm tries to maintain constant relative step size while respecting
      absolute step size constraints
    - When constraints are active, the spacing deviates from pure geometric progression
    - The final point is adjusted to exactly match the stop value
    - More points may be generated than in pure geometric spacing due to constraints

    Examples
    --------
    >>> # Basic geometric-like spacing
    >>> seq = constrained_geomspace(2, 100, relative_step=0.2)
    >>> print(seq)

    >>> # With minimum step constraint (useful for small values)
    >>> seq = constrained_geomspace(1, 50, relative_step=0.1, min_step=1.0)
    >>> print(seq)

    >>> # With maximum step constraint (useful for large values)
    >>> seq = constrained_geomspace(10, 1000, relative_step=0.3, max_step=50)
    >>> print(seq)

    >>> # Compare with your smart_grid_smooth
    >>> smart_seq = smart_grid_smooth(2, 300, min_step=2, max_step=20)
    >>> constrained_seq = constrained_geomspace(2, 300, relative_step=0.15, min_step=2, max_step=20)
    """
    if start <= 0:
        raise ValueError("start must be positive")
    if stop <= start:
        raise ValueError("stop must be greater than start")
    if relative_step <= 0:
        raise ValueError("relative_step must be positive")

    # Use numpy for computation, convert to jax at the end if needed
    values = [start]
    current = start

    while current < stop:
        # Calculate ideal next step based on relative step size
        ideal_step = current * relative_step

        # Apply constraints
        if min_step is not None:
            ideal_step = max(ideal_step, min_step)
        if max_step is not None:
            ideal_step = min(ideal_step, max_step)

        # Calculate next value
        next_value = current + ideal_step

        # Don't overshoot the target
        if next_value >= stop:
            values.append(stop)
            break
        else:
            values.append(next_value)
            current = next_value

    # Convert to appropriate array type
    result = np.array(values)
    return jnp.array(result) if use_jax else result


def adaptive_geomspace(
    start: float,
    stop: float,
    target_points: int = None,
    relative_step: float = None,
    min_step: float = None,
    max_step: float = None,
    use_jax: bool = True,
) -> Union[np.ndarray, jnp.ndarray]:
    """
    Generate geometric-like spacing with either target number of points or relative step size.

    This function automatically determines the relative step size to achieve approximately
    the target number of points, while respecting step size constraints.

    Parameters
    ----------
    start, stop : float
        Start and end values
    target_points : int, optional
        Desired approximate number of points. If provided, relative_step is ignored.
    relative_step : float, optional
        Target relative step size. Used only if target_points is None.
    min_step, max_step : float, optional
        Step size constraints
    use_jax : bool, optional
        Whether to return JAX array

    Returns
    -------
    sequence : jnp.ndarray or np.ndarray
        Array of values with constrained spacing

    Examples
    --------
    >>> # Generate approximately 50 points
    >>> seq = adaptive_geomspace(2, 400, target_points=50, min_step=1, max_step=25)

    >>> # Use specific relative step
    >>> seq = adaptive_geomspace(2, 400, relative_step=0.1, min_step=1, max_step=25)
    """
    if target_points is not None:
        # Estimate relative step size to get approximately target_points
        if min_step is None and max_step is None:
            # Pure geometric case - can calculate exactly
            estimated_relative_step = (stop / start) ** (1 / (target_points - 1)) - 1
        else:
            # With constraints - use iterative approach to estimate
            estimated_relative_step = 0.1  # Initial guess

            # Binary search for good relative step size
            low_step, high_step = 0.001, 1.0
            for _ in range(20):  # Max iterations
                test_seq = constrained_geomspace(
                    start,
                    stop,
                    estimated_relative_step,
                    min_step,
                    max_step,
                    use_jax=False,
                )
                n_points = len(test_seq)

                if abs(n_points - target_points) <= 1:
                    break
                elif n_points < target_points:
                    high_step = estimated_relative_step
                    estimated_relative_step = (low_step + estimated_relative_step) / 2
                else:
                    low_step = estimated_relative_step
                    estimated_relative_step = (estimated_relative_step + high_step) / 2

        return constrained_geomspace(
            start, stop, estimated_relative_step, min_step, max_step, use_jax
        )

    elif relative_step is not None:
        return constrained_geomspace(
            start, stop, relative_step, min_step, max_step, use_jax
        )

    else:
        raise ValueError("Either target_points or relative_step must be provided")


# Convenience function that mimics your smart_grid_smooth behavior
def smart_geomspace(
    start: float = 5,
    stop: float = 300,
    min_step: float = 2,
    max_step: float = 20,
    relative_step: float = 0.15,
    use_jax: bool = True,
) -> Union[np.ndarray, jnp.ndarray]:
    """
    Drop-in replacement for smart_grid_smooth with geometric-like behavior.

    This function provides similar functionality to your smart_grid_smooth but with
    more predictable geometric-like spacing.

    Parameters
    ----------
    start, stop : float
        Start and end values
    min_step, max_step : float
        Minimum and maximum step sizes
    relative_step : float, optional
        Target relative step size (default 0.15 = 15% increase per step)
    use_jax : bool, optional
        Whether to return JAX array

    Returns
    -------
    sequence : jnp.ndarray or np.ndarray
        Spacing similar to smart_grid_smooth but more systematic

    Examples
    --------
    >>> # Replace your smart_grid_smooth calls with:
    >>> sizes = smart_geomspace(2, 399.9, min_step=2, max_step=25)
    >>> sizes = smart_geomspace(2, 350, min_step=2, max_step=20)
    """
    return constrained_geomspace(
        start, stop, relative_step, min_step, max_step, use_jax
    )


def smooth_step(x, frac=0.1, min_step=2, max_step=10):
    """Smooth step function, I hope to more optimally cover size space this way, to have smaller steps at small sizes and larger steps at large sizes. and minimal amount of steps overall"""

    raw = x * frac
    return min_step + (max_step - min_step) * (
        jnp.arctan((raw - min_step) / 2) / jnp.pi + 0.5
    )


def smart_grid_smooth(start=5, stop=300, min_step=2, max_step=10):
    values = [start]
    while values[-1] < stop:
        step = smooth_step(values[-1], min_step=min_step, max_step=max_step)
        values.append(values[-1] + step)
    return jnp.array(values)
