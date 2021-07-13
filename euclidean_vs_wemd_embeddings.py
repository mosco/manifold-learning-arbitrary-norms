import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy
import scipy.spatial.distance
import sklearn.metrics

import ATP_synthase_datagen
import wemd
import laplacian
import pickler
from timer import Timer


FIGURES_DIR = 'figures'
N_RANGE = [25,50,100,200,400,800]
#N_RANGE = [25,50]
STD = 0.01644027 
WAVELET_THRESHOLD_MASS_PERCENT_LIST = [0.9, 1.0]

SEED = 2020
MARKERSIZE = 75
MARKEREDGEWIDTH = 0.5 
DPI = 300
EXTENSION = 'pdf'


def picklename_dataset(n):
    return f'rotated_ATP_syntase_{n}'


def picklename_dataset_metadata(n):
    return f'rotated_ATP_syntase_metadata_{n}'


def picklename_pairwise_distances(n, std, wavelet_threshold_mass_percent):
    return f'distances_n={n}_std={std}_WaveThreshMassPercent={float(wavelet_threshold_mass_percent)}'


def generate_datasets(n_range):
    np.random.seed(SEED)
    for n in n_range:
        print(f'    n={n}',)
        (data, angles) = ATP_synthase_datagen.build_dataset(n)

        cmap = plt.get_cmap("hsv")
        colors = [matplotlib.colors.to_hex(cmap(angle/360)) for angle in angles]

        pickler.dump(picklename_dataset(n), volumes=data, angles=angles, colors=colors)
        pickler.dump(picklename_dataset_metadata(n), angles=angles, colors=colors)


def all_l1_distances_dense(X):
    assert type(X) == np.ndarray
    return scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(X, 'cityblock')) 


def all_l1_distances_sparse(X):
    assert type(X) == scipy.sparse.csr_matrix
    return sklearn.metrics.pairwise_distances(X, metric='l1', n_jobs=4)


def all_l2_distances(X):
    # Surisingly, it is faster with n_jobs=1
    #return sklearn.metrics.pairwise_distances(X, metric='l2', n_jobs=1) 

    #SINGLE CORE
    return scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(X, 'euclidean')) 


def flatten_last_dimensions(arr):
    return arr.reshape((arr.shape[0], np.prod(arr.shape[1:])))


def find_threshold(arr, percentile):
    assert arr.ndim == 1
    assert 0 <= percentile <= 1

    arr_sorted = np.flip(np.sort(arr))
    cumsum_large_enough_mask = np.cumsum(arr_sorted, dtype=np.float64) > (percentile*np.sum(arr_sorted, dtype=np.float64))
    indices = np.where(cumsum_large_enough_mask)[0]
    if len(indices) == 0:
        return arr_sorted[0]
    else:
        return arr_sorted[indices[0]]


def sparsify_array(dense_array, threshold, prints=False):
    """
    Return a sparse csr_array that contains only the entries of dense_array
    whose absolute value is >= threshold.
    """
    absolute_dense_array = np.absolute(dense_array)
    (rows, cols) = np.where(absolute_dense_array >= threshold)
    sparse_array = scipy.sparse.csr_matrix((dense_array[(rows, cols)], (rows, cols)), shape=dense_array.shape)

    if prints:
        n_full_coefs = np.prod(dense_array.shape)
        n_sparse_coefs = sparse_array.nnz
        print(f'Threshold: {threshold}')
        print(f'Absolute coefficient sum before threshold: {absolute_dense_array.sum()}')
        print(f'Post-threshold coefficient number: {n_sparse_coefs} / {n_full_coefs} ({n_sparse_coefs/n_full_coefs})')
        print(f'Absolute coefficient sum after threshold: {np.absolute(sparse_array).sum()}')

    return sparse_array


def precalc_pairwise_distances(wavelet, level, std, wavelet_threshold_mass_percent):
    assert 0 <= wavelet_threshold_mass_percent <= 1
    print(f'Computing distances. wavelet={wavelet}, level={level}, std={std}')

    l2_times = []
    l1_times = []
    dwt_times = []
    sparsification_times = []

    threshold = None

    for n in N_RANGE:
        print(f'\n---- {n} -------------------------------------------------\n')
        print(f'Loading {picklename_dataset(n)}')
        d = pickler.load(picklename_dataset(n))
        noisy_volumes = d.volumes + np.random.normal(loc=0, scale=std, size=d.volumes.shape)
        flattened_noisy_volumes = flatten_last_dimensions(noisy_volumes)

        with Timer('Computing pairwise l2 distances') as timer_l2:
            l2_distances = all_l2_distances(flattened_noisy_volumes)
        l2_times.append(timer_l2.elapsed)

        with Timer('Computing weighted wavelet transforms for all volumes') as timer_dwt:
            wemd_embeddings = np.array([wemd.embed(vol, wavelet, level) for vol in noisy_volumes])
        dwt_times.append(timer_dwt.elapsed)

        if wavelet_threshold_mass_percent == 1.0:
            print('(using dense matrices)')
            with Timer('Computing pairwise l1 distances') as timer_l1:
                wemd_distances = all_l1_distances_dense(wemd_embeddings)
            l1_times.append(timer_l1.elapsed)
        else:
            print('(using sparse matrices)')
            with Timer('Sparsifying wemd results') as timer_sparsification:
                processed_wemd_embeddings = wemd_embeddings - np.mean(wemd_embeddings, axis=0)
                if threshold is None: # Compute threshold only on the smallest dataset with n = n_range[0]
                    threshold = find_threshold(np.absolute(processed_wemd_embeddings).flatten(), wavelet_threshold_mass_percent)
                sparsified_wemds = sparsify_array(processed_wemd_embeddings, threshold, prints=True)
            sparsification_times.append(timer_sparsification.elapsed)

            with Timer('Computing pairwise l1 distances') as timer_l1:
                wemd_distances = all_l1_distances_sparse(sparsified_wemds)
            l1_times.append(timer_l1.elapsed)

        pickler.dump(picklename_pairwise_distances(n, std, wavelet_threshold_mass_percent), l2_distances=l2_distances, wemd_distances=wemd_distances)
    
    print(f'l2_times: {l2_times}')
    print(f'dwt_times: {dwt_times}')
    print(f'sparsification_times: {sparsification_times}')
    print(f'l1_times: {l1_times}')
    pickler.dump(f'runtimes_std={std}_WaveThreshMassPercent={wavelet_threshold_mass_percent}'.replace('.','_'), n_range=N_RANGE, l2_times=l2_times, dwt_times=dwt_times, l1_times=l1_times, sparsification_times=sparsification_times, wavelet=wavelet, level=level)


def leading_eigenvectors(L_normalized, n_eigenvectors):
    assert laplacian.is_symmetric(L_normalized)
    assert L_normalized.dtype == float

    (eigvals, eigvecs) = scipy.sparse.linalg.eigs(L_normalized, n_eigenvectors, which='SM')
    return (eigvals.real, eigvecs.real.transpose())


def gaussian_kernel(distance_matrix, sigma):
    assert distance_matrix.ndim == 2
    assert distance_matrix.shape[0] == distance_matrix.shape[1]

    m = np.exp(-distance_matrix.astype(float)**2/(2*sigma**2))
    np.fill_diagonal(m, 0)

    return m

def epsilon_kernel(distance_matrix, epsilon):
    assert distance_matrix.ndim == 2
    assert distance_matrix.shape[0] == distance_matrix.shape[1]

    m = (distance_matrix < epsilon)
    np.fill_diagonal(m, 0)

    return m

def plot_euclidean_embedding(n, std, kernel, sigma, markersize, alpha, figures_dir):
    WL2 = pickler.load(picklename_pairwise_distances(n, std, wavelet_threshold_mass_percent=1.0)).l2_distances
    colors = pickler.load(picklename_dataset_metadata(n)).colors
    #W = gaussian_kernel((WL2+WL2.T)/2.0, sigma)
    W = kernel((WL2+WL2.T)/2.0, sigma)
    L = laplacian.combinatorial(W)
    (eigvals, eigvecs) = np.linalg.eig(L)
    eigvecs = np.array(eigvecs)
    plt.figure(figsize=(1.688, 1.688), dpi=DPI, frameon=False)
    plt.axis('off')
    plt.scatter(eigvecs[:,1], eigvecs[:,2], s=markersize, c=colors, edgecolors='black', linewidth=MARKEREDGEWIDTH, alpha=alpha)

    fn = os.path.join(figures_dir, f'euclidean_embedding_n={n}_std={std}_seed={SEED}_{kernel.__qualname__}'.replace('.','_') + f'.{EXTENSION}')
    print('Saving', fn)
    plt.savefig(fn, bbox_inches='tight', pad_inches=0)


def plot_wemd_embedding(n, std, wavelet_threshold_mass_percent, kernel, sigma, markersize, alpha, figures_dir):
    Wemd = pickler.load(picklename_pairwise_distances(n, std, wavelet_threshold_mass_percent)).wemd_distances
    colors = pickler.load(picklename_dataset_metadata(n)).colors
    #W = gaussian_kernel((Wemd+Wemd.T)/2.0, sigma)
    W = kernel((Wemd+Wemd.T)/2.0, sigma)
    L = laplacian.combinatorial(W)
    (eigvals, eigvecs) = np.linalg.eig(L)
    eigvecs = np.array(eigvecs)
    plt.figure(figsize=(1.688, 1.688), dpi=DPI, frameon=False)
    plt.axis('off')
    plt.scatter(eigvecs[:,1], eigvecs[:,2], s=markersize, c=colors, edgecolors='black', linewidth=MARKEREDGEWIDTH, alpha=alpha)

    fn = os.path.join(figures_dir, f'wemd_embedding_n={n}_std={std}_seed={SEED}_{kernel.__qualname__}_threshold={wavelet_threshold_mass_percent}'.replace('.','_') + f'.{EXTENSION}')
    print('Saving', fn)
    plt.savefig(fn, bbox_inches='tight', pad_inches=0)


def precalc_all(wavelet, level):
    print('\n==== Generating datasets ========================================================================\n')
    generate_datasets(N_RANGE)

    for wavelet_threshold_mass_percent in WAVELET_THRESHOLD_MASS_PERCENT_LIST:
        print('\n=================================================================================================')
        print('==== Computing pairwise distances (noiseless) ===================================================')
        print(f'         Wavelet threshold mass percent: {wavelet_threshold_mass_percent} ')
        print('=================================================================================================')
        precalc_pairwise_distances(wavelet, level, 0, wavelet_threshold_mass_percent)

        print('\n=================================================================================================')
        print('==== Computing pairwise distances (noisy) =======================================================')
        print(f'         Wavelet threshold mass percent: {wavelet_threshold_mass_percent} ')
        print('=================================================================================================')
        precalc_pairwise_distances(wavelet, level, STD, wavelet_threshold_mass_percent)


def plot_all_gaussian_kernel(figures_dir):
    # Gaussian kernel width. For both the Euclidean and WEMD with the unweighted Laplacian, anything above 100 gives similar results.
    ALPHA = 0.2
    MARKERSIZE = 80
    EUCLIDEAN_SIGMA_DICT = {25:8, 50:6, 100:4, 200:3.5, 400:3, 800:3} # For Gaussian kernel
    for std in (0, STD):
        print(f'==== Generating Euclidean embeddings (noise std={std} ============')
        for n in N_RANGE:
            sigma = EUCLIDEAN_SIGMA_DICT[n]
            plot_euclidean_embedding(n, std, gaussian_kernel, sigma, MARKERSIZE, ALPHA, figures_dir)
        print('')

        for wavelet_threshold_mass_percent in WAVELET_THRESHOLD_MASS_PERCENT_LIST:
            print(f'==== Generating WEMD embeddings (noise std={std} wavelet_threshold_mass_percent={wavelet_threshold_mass_percent} =================')
            for n in N_RANGE:
                sigma = 30 # Gaussian kernel
                plot_wemd_embedding(n, std, wavelet_threshold_mass_percent, gaussian_kernel, sigma, MARKERSIZE, ALPHA, figures_dir)
            print('')


def plot_all_epsilon_kernel(figures_dir):
    # Gaussian kernel width. For both the Euclidean and WEMD with the unweighted Laplacian, anything above 100 gives similar results.
    ALPHA = 0.2
    MARKERSIZE = 80
    print(f'==== Generating Euclidean embeddings (noiseless) ============')
    for n in N_RANGE:
        sigma = 8 # For epsilon-kernel
        plot_euclidean_embedding(n, 0, epsilon_kernel, sigma, MARKERSIZE, ALPHA, figures_dir)
    print('')

    WEMD_SIGMA_DICT = {25:32, 50:32, 100:32, 200:24, 400:24, 800:24}
    for wavelet_threshold_mass_percent in WAVELET_THRESHOLD_MASS_PERCENT_LIST:
        print(f'==== Generating WEMD embeddings (noiseless, wavelet_threshold_mass_percent={wavelet_threshold_mass_percent} =================')
        for n in N_RANGE:
            sigma = WEMD_SIGMA_DICT[n]
            plot_wemd_embedding(n, 0, wavelet_threshold_mass_percent, epsilon_kernel, sigma, MARKERSIZE, ALPHA, figures_dir)
        print('')

