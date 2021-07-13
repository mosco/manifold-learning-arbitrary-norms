"""Dataset for roughly emulating ATP-synthase rotational configurations (not biologically accurate).
The volumetric data was generated from Protein Data Bank entry 1E79.
"""
import multiprocessing
from numpy import array, remainder, pi
from numpy.random import default_rng
import scipy.ndimage as ndimage

import mrcfile

from timer import Timer

N_CPU = multiprocessing.cpu_count()

ATP_SYNTHASE_SHAFT3D_FILENAME = 'rotating_shaft_res6.mrc'
ATP_SYNTHASE_SHAFT3D_SHAPE = (47, 104, 47)


def _generate_random_angle_deg(rng):
    """
    ATP-Synthase acts as a stepper motor, spending most of its time in 3 preferred orientations.
    """
    angle_deg = rng.choice([0, 120, 240, rng.uniform(0,360)], p=[0.2, 0.2, 0.2, 0.4])
    jitter_deg = rng.normal(0, 1)
    return remainder(angle_deg + jitter_deg, 360)


def _rotate_shaft(shaft3D, angle_deg):
    assert shaft3D.shape == ATP_SYNTHASE_SHAFT3D_SHAPE
    rotated_shaft3D = ndimage.rotate(shaft3D, angle_deg, axes=(0,2), reshape=False)
    assert rotated_shaft3D.shape == ATP_SYNTHASE_SHAFT3D_SHAPE
    return rotated_shaft3D


def build_dataset(n, angles_deg=None, seed=289742):
    with mrcfile.open(ATP_SYNTHASE_SHAFT3D_FILENAME) as mrc:
        shaft3D = mrc.data
    assert shaft3D.shape == ATP_SYNTHASE_SHAFT3D_SHAPE

    rng = default_rng(seed)
    if angles_deg is None:
        angles_deg = sorted([_generate_random_angle_deg(rng) for i in range(n)])
    assert len(angles_deg) == n

    with Timer(f'Computing {n} ATP Synthase rotations (N_CPU={N_CPU})'):
        args = [(shaft3D, angle) for angle in angles_deg]
        with multiprocessing.Pool(processes=N_CPU) as pool:
            rotated_shafts = pool.starmap(_rotate_shaft, args)

    return (array(rotated_shafts), angles_deg)

