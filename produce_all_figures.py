#!/usr/bin/env python3
import os
import matplotlib.pyplot as plt
import numpy as np

import ATP_synthase_datagen
import euclidean_vs_wemd_metrics
import euclidean_vs_wemd_embeddings
import weighted_L1_eigenfunctions_on_circle

WAVELET = 'sym3'
LEVEL = 6

FIGURES_DIR = 'figures'

def figure_rotor_slice(figures_dir):
        VMIN =-0.06 
        VMAX = 0.150

        (volumes, angles) = ATP_synthase_datagen.build_dataset(1, angles_deg=[0])
        m = volumes[0]

        plt.figure()
        plt.imshow(m[25,::-1,:], vmin=VMIN, vmax=VMAX)
        fn = os.path.join(figures_dir, 'slice_noiseless.png')
        print('Saving', fn)
        plt.savefig(fn, bbox_inches='tight', pad_inches=0)

        plt.figure()
        noisy_m = m + np.random.normal(loc=0, scale=euclidean_vs_wemd_embeddings.STD, size=m.shape)
        plt.imshow(noisy_m[25,::-1,:], vmin=VMIN, vmax=VMAX)
        fn = os.path.join(figures_dir, 'slice_noisy.png')
        print('Saving', fn)
        plt.savefig(fn, bbox_inches='tight', pad_inches=0)


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)

    print('==== Saving rotor slices =========================================================')
    figure_rotor_slice(FIGURES_DIR)

    print('= Computing and saving WEMD vs Euclidean figure')
    euclidean_vs_wemd_metrics.compute_and_plot(WAVELET, LEVEL, FIGURES_DIR)

    print(' ==== Running all WEMD embedding calculations ====')
    euclidean_vs_wemd_embeddings.precalc_all(WAVELET, LEVEL)
    print(' ==== Producing all WEMD embedding figures ====')
    euclidean_vs_wemd_embeddings.plot_all_gaussian_kernel(FIGURES_DIR)

    print(' ==== Computing and plotting all sanity check figures for the empirical L1 Laplacian ====')
    weighted_L1_eigenfunctions_on_circle.plot_all_check_L1_laplacian()
    print(' ==== Computing and plotting first eigenvectors of the weighted L1 norm Laplacian on the circle')
    weighted_L1_eigenfunctions_on_circle.plot_all_first_eigenvectors()
    

if __name__ == '__main__':
    main()
