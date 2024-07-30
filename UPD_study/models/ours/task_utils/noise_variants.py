from itertools import product

import numpy as np


def interpolant(t):
    return t*t*t*(t*(t*6 - 15) + 10)


def generate_perlin_noise(shape, num_intervals, tileable=False, interpolant=interpolant):
    """Generate a numpy array of perlin noise.

    Args:
        shape: The shape of the generated array (tuple of ints).
        num_intervals: The number of periods of noise to generate along each
            axis (tuple of ints).
        tileable: If the noise should be tileable along each axis
            (bool or tuple of bools). Defaults to False.
        interpolant: The interpolation function, defaults to the
            smootherstep function: t*t*t*(t*(t*6 - 15) + 10).

    Returns:
        A numpy array with the generated noise.

    Raises:
        ValueError: If tileable=True and shape is not a multiple of num_intervals.
    """
    if not isinstance(shape, tuple):
        raise ValueError(f"shape must be a tuple of integers: {shape}")

    if not isinstance(num_intervals, tuple):
        raise ValueError(
            f"num_intervals must be a tuple of floats: {num_intervals}")

    if len(shape) != len(num_intervals):
        raise ValueError(
            f"shape and num_intervals must have the same length: {shape}, {num_intervals}")

    if isinstance(tileable, tuple) and len(tileable) != len(shape):
        raise ValueError(
            "tileable must have the same length as shape when given as a tuple.")

    # Gradients
    gradients_shape = tuple(int(np.ceil(r + 1)) for r in num_intervals)
    vector_points = np.random.standard_normal(gradients_shape + (len(shape),))
    gradients = vector_points / \
        np.linalg.norm(vector_points, axis=-1, keepdims=True)

    if isinstance(tileable, bool):
        tileable = (tileable,) * len(shape)

    for axis, is_tileable in enumerate(tileable):
        if is_tileable:

            if shape[axis] % num_intervals[axis] != 0:
                raise ValueError("When tileable, shape[axis] must be a multiple of num_intervals[axis]:"
                                 f"shape[axis]={shape[axis]}, num_intervals[axis]={num_intervals[axis]}")

            front_slices = [slice(None)] * len(gradients_shape)
            front_slices[axis] = 0

            back_slices = [slice(None)] * len(gradients_shape)
            back_slices[axis] = -1
            gradients[tuple(back_slices)] = gradients[0][tuple(front_slices)]

    # Create the grid
    grid = np.meshgrid(*[np.arange(0, s) * r / s for s,
                       r in zip(shape, num_intervals)], indexing='ij')
    grid = np.stack(grid, axis=-1)

    # Calculate the indices
    grid_floor = np.floor(grid).astype(int)
    grid_ceil = np.ceil(grid).astype(int)

    coord_pairs = [((0, grid_floor[..., i]), (1, grid_ceil[..., i]))
                   for i in range(len(shape))]

    corner_dot_tuples = []

    # Just to avoid calculating it twice
    grid_frac = None

    for corner_slice_tuples in product(*coord_pairs):
        corner_code, corner_slices = zip(*corner_slice_tuples)

        # Equivalent to gNNN
        corner_vectors = gradients[corner_slices]

        # Corner coordinates stacked into single numpy array
        corner_coords = np.stack(corner_slices, axis=-1)

        # Rather than getting fractional relative to grid_floor and manipulating it
        # for each corner, directly calculate the fractional relative to each corner
        grid_from_corner = grid - corner_coords

        if all([c == 0 for c in corner_code]):
            grid_frac = grid_from_corner

        # Dot product of gradient and grid_from_corner
        corner_dot_tuples.append((corner_code, np.sum(
            corner_vectors * grid_from_corner, axis=-1)))

    assert grid_frac is not None

    # Interpolate adjacent corners until we have the final value
    corner_dot_tuples.sort(key=lambda x: x[0])
    assert len(corner_dot_tuples) == 2**len(shape)

    t = interpolant(grid_frac)

    for i in range(len(shape)):

        # For each dimension, we get the current values for the beginning and end along that direction

        first_half_codes, first_half_dots = zip(
            *corner_dot_tuples[:len(corner_dot_tuples)//2])
        assert all([c[0] == 0 for c in first_half_codes])

        second_half_codes, second_half_dots = zip(
            *corner_dot_tuples[len(corner_dot_tuples)//2:])
        assert all([c[0] == 1 for c in second_half_codes])

        assert all([c1[1:] == c2[1:]
                   for c1, c2 in zip(first_half_codes, second_half_codes)])

        # Interpolate values using the smootherstep function

        curr_dim_interp = t[..., i]
        interpolated_dots = first_half_dots * \
            (1 - curr_dim_interp) + second_half_dots * curr_dim_interp

        # New codes are the same as the old codes, but with the first element removed
        # As we have interpolated along that dimension
        new_codes = [c[1:] for c in first_half_codes]

        # Construct new corner_dot_tuples so we can repeat along next dimension
        corner_dot_tuples = list(zip(new_codes, interpolated_dots))

    assert len(corner_dot_tuples) == 1

    final_code, final_dot = corner_dot_tuples[0]

    # Final code should be an empty tuple, as we have interpolated all the way down to a single point
    assert final_code == tuple()

    assert final_dot.shape == shape

    return final_dot


def generate_fractal_noise(shape, all_num_intervals, amplitudes, tileable=False,
                           interpolant=interpolant):
    """Generate a numpy array of fractal noise.

    Args:
        shape: The shape of the generated array (tuple of ints).
        all_num_intervals: For each noise scale, the number of periods of noise to
            generate along each axis (list of tuple of ints).
        amplitudes: The amplitude of each noise scale (list of floats).
        tileable: If the noise should be tileable along each axis
            (bool or tuple of bools). Defaults to False.
        interpolant: The interpolation function, defaults to the
            smootherstep function: t*t*t*(t*(t*6 - 15) + 10).

    Returns:
        A numpy array with the generated noise.

    Raises:
        ValueError: If tileable=True and shape is not a multiple of num_intervals.
    """

    assert len(all_num_intervals) == len(amplitudes)

    f_noise = None

    for num_intervals, amplitude in zip(all_num_intervals, amplitudes):
        p_noise = generate_perlin_noise(
            shape, num_intervals, tileable, interpolant)
        if f_noise is None:
            f_noise = amplitude * p_noise
        else:
            f_noise += amplitude * p_noise

    return f_noise
