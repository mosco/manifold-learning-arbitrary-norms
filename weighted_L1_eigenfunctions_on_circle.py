import os

import numpy as np
from numpy import pi, sin, cos, absolute, linspace, sign, mean, hstack, vstack, argsort, transpose, zeros, asarray, exp
from scipy.special import gamma

import scipy.sparse.linalg
import sklearn.neighbors

import matplotlib
import matplotlib.pyplot as plt

import laplacian
import pickler
from latex_fig_utils import RCPARAMS_LATEX_DOUBLE_COLUMN, save_figure


EPSILON = 1e-18


def periodic1D_first_derivative(n):
    delta_x = 2*pi / n
    return scipy.sparse.diags([[0.5]*1, [-0.5]*(n-1), [0.5]*(n-1), [-0.5]*1], [-(n-1), -1, 1, n-1]) / (delta_x)


def periodic1D_second_derivative(n):
    delta_x = 2*pi / n
    return scipy.sparse.diags([[1]*1, [1]*(n-1), [-2]*n, [1]*(n-1), [1]*1], [-(n-1), -1, 0, 1, n-1]) / (delta_x**2)


def weighted_L1_laplacian_wrong(W1, W2, resolution):
    L = periodic1D_second_derivative(resolution)

    X = linspace(0, 2*pi, resolution+1)[:-1]
    reweighting_matrix = scipy.sparse.diags([1.0 / (3*(absolute(W1*sin(X)) + absolute(W2*cos(X)))**3)], [0])

    return (X, reweighting_matrix*L)


def weighted_L1_laplacian_new(W1, W2, resolution):
    dx = periodic1D_first_derivative(resolution)
    dxx = periodic1D_second_derivative(resolution)

    X = linspace(0, 2*pi, resolution+1)[:-1]

    S = sign(sin(X)*cos(X))
    B = -W1*absolute(cos(X)) + W2*absolute(sin(X))
    C = W1*absolute(sin(X)) + W2*absolute(cos(X))


    dx_weights =  S*B/(C**4)
    dx_weights_diagonalized = scipy.sparse.diags([dx_weights], [0])

    dxx_weights = +1/(3*(C**3))
    dxx_weights_diagonalized = scipy.sparse.diags([dxx_weights], [0])

    weighted_L1_laplacian = dx_weights_diagonalized*dx + dxx_weights_diagonalized*dxx

    return (X, weighted_L1_laplacian)


def compute_eigvalsvecs(L, n_eigvecs):

    (eigvals, eigvecs_transposed) = scipy.sparse.linalg.eigs(L, k=n_eigvecs, sigma=1, which='LM')

    # # We are solving the problem Lf + \lambda \rho f = 0
    # # which corresponds to Lf = -\lambda \rho f
    # # so in our convention we need to flip the sign of \lambda
    # eigvals = -eigvals

    #print(f'eigvals: {eigvals}')
    # The eigenvalues are supposed to be close to real, since the matrix is almost-symmetric.
    # Sturm-Liouville also says that the eigenvalues (in the continuous case) should be real.
    assert (np.absolute(eigvals.imag) < 0.001).all()
    eigvals = eigvals.real
    sorted_eigval_indices = argsort(eigvals)[::-1]
    eigvals = eigvals.real[sorted_eigval_indices]

    # Likewise, Sturm-Liouville theory says that the eigenvectors can be chosen to be real.
    # It is interesting that (for large enough) resoltions they are indeed real.
    print(f' np.absolute(eigvecs_transposed.imag).max(): { np.absolute(eigvecs_transposed.imag).max()}')
    assert (np.absolute(eigvecs_transposed.imag) == 0).all()
    eigvecs = (eigvecs_transposed.T)[sorted_eigval_indices].real

    return (eigvals, eigvecs)


def plot_unitcircle_eigvecs(eigvals, eigvecs, figure_filename, eigenvalue_legend_n_digits):
    (resolution,) = eigvecs[0].shape
    X = linspace(0, 2*pi, resolution+1)

    rc_dbl_col = RCPARAMS_LATEX_DOUBLE_COLUMN
    with matplotlib.rc_context(rc = rc_dbl_col):

        wide_figure_size = tuple([2,1] * np.array(RCPARAMS_LATEX_DOUBLE_COLUMN['figure.figsize']))
        fig, ax = plt.subplots(figsize = wide_figure_size)

        for (eigval, eigvec) in zip(eigvals, eigvecs): 
            Y = hstack((eigvec, eigvec[0]))
            if +0.0 <= eigval < EPSILON:
                eigval = -eigval
            ax.plot(X, Y, '-', label=f'$\\lambda={eigval:.{eigenvalue_legend_n_digits}f}$') 

        ax.legend(loc='lower left')
        ax.set_xticks([0, pi/2, pi, 1.5*pi, 2*pi])
        ax.set_xlim([0, 2*pi])
        ax.set_xlabel(r'$\theta$')
        ax.set_ylabel(r'$\varphi(\theta)$')
        ax.set_xticklabels([r'$0$', r'$\frac{\pi}{2}$', r'$\pi$', r'$\frac{3}{2} \pi$', r'$2 \pi$'])
        ax.set_yticks([0])
        ax.grid()

        if figure_filename is not None:
            save_figure(fig, figure_filename)
            plt.close(fig)


def plot_first_eigenvectors(W1, W2, resolution, n_eigvecs, eigenvalue_legend_n_digits):
    assert resolution > 10
    (X, L) = weighted_L1_laplacian_new(W1, W2, resolution)
    (eigvals, eigvecs) = compute_eigvalsvecs(L, n_eigvecs)

    # Eigenvectors have a sign-ambiguity. For consistency of plots we take the positive-mean choice.
    for (i, eigvec) in enumerate(eigvecs):
        eigvecs[i] *= sign(eigvec[10])

    plot_unitcircle_eigvecs(eigvals, eigvecs, f'weighted_L1_eigvecs_W1={W1}_W2={W2}_resolution={resolution}_numeigvecs={n_eigvecs}', eigenvalue_legend_n_digits)

    return (eigvals, eigvecs)


def epsilon_ball_L1_laplacian_at(i, points, epsilon, w1, w2):
    n = len(points)
    assert points.shape == (n,2)

    diffs = points - points[i]
    assert diffs.shape == (n,2)

    W = (w1*absolute(diffs[:,0]) + w2*absolute(diffs[:,1]) < epsilon).astype(int)
    D = zeros(n)
    D[i] = W.sum()
    return W-D

# Check that the eigenvectors match and that the convolution of the matrix rows with x^w looks the same


def empirical_laplacian_gaussian_kernel(w1, w2, n, sigma, radius):
    #point_angles = linspace(0,2*pi,n+1)[:n]
    point_angles = 2*pi*np.random.random(n)
    point_angles.sort()
    points = transpose(vstack((cos(point_angles), sin(point_angles))))
    points[:,0] *= w1
    points[:,1] *= w2
    Lhat = zeros((n,n))
    W = sklearn.neighbors.radius_neighbors_graph(points, radius=radius, mode='distance', include_self=False, p=1)
    W.data = np.e**(-W.data**2 / (sigma**2))
    return (point_angles, -laplacian.combinatorial(W))


def empirical_laplacian_epsilon_kernel(w1, w2, n, sigma):
    point_angles = linspace(0,2*pi,n+1)[:n]
    #point_angles = 2*pi*np.random.random(n)
    point_angles.sort()
    points = transpose(vstack((cos(point_angles), sin(point_angles))))
    points[:,0] *= w1
    points[:,1] *= w2
    Lhat = zeros((n,n))
    W = sklearn.neighbors.radius_neighbors_graph(points, radius=sigma, mode='connectivity', include_self=False, p=1)
    return laplacian.combinatorial(W)


def empirical_epsilon_laplacian_num_points(w1, w2, n, epsilon):
    X = linspace(0,2*pi,n+1)[:n]
    #point_angles = 2*pi*np.random.random(n)
    X.sort()
    points = transpose(vstack((cos(X), sin(X))))
    points[:,0] *= w1
    points[:,1] *= w2
    Lhat = zeros((n,n))
    W = sklearn.neighbors.radius_neighbors_graph(points, radius=epsilon, mode='connectivity', include_self=False, p=1)
    Y = asarray(W.sum(axis=1).astype(int)).flatten()
    return (X,Y)


def empirical_gaussian_laplacian_num_points(w1, w2, n, sigma):
    X = linspace(0,2*pi,n+1)[:n]
    #point_angles = 2*pi*np.random.random(n)
    X.sort()
    points = transpose(vstack((cos(X), sin(X))))
    points[:,0] *= w1
    points[:,1] *= w2
    Lhat = zeros((n,n))
    W = sklearn.neighbors.radius_neighbors_graph(points, radius=sigma*4, mode='distance', include_self=False, p=1)

    # 1/(4s^2) is the correct normalization for the kernel e^(-(|x|+|y|)^2/(2s^2))
    W.data = np.e**(-W.data**2 / (2*sigma**2))
    Y = asarray(W.sum(axis=1)).flatten()
    return (X,Y)


def plot_empirical_gaussian_laplacian_num_points(w1, w2, n, epsilon):
    (X,Y) = empirical_gaussian_laplacian_num_points(w1, w2, n, epsilon)

    fig, ax = plt.subplots()
    ax.set_xticks([0, pi/2, pi, 1.5*pi, 2*pi])
    ax.set_xlim([0, 2*pi])
    ax.set_xlabel(r'$\theta$')
    ax.set_ylabel(r'num points')
    ax.set_xticklabels([r'$0$', r'$\frac{\pi}{2}$', r'$\pi$', r'$\frac{3}{2} \pi$', r'$2 \pi$'])
    # Integral of e^(-(|x|)^2/(2*s^2))dx from -infinity to +infinity is sqrt(2 pi)s
    # Not sure why the 2 factor is needed.
    plt.plot(X, Y*2/np.sqrt(2*pi), '-', label='Empirical')
    ax.grid()

    line_segment_length = 2*epsilon/(w1*absolute(sin(X)) + w2*absolute(cos(X)))
    Ytheory = n*line_segment_length/(2*pi)
    plt.plot(X, Ytheory, '--', label='Theory')
    plt.legend()


def plot_empirical_epsilon_laplacian_num_points(w1, w2, n, epsilon):
    (X,Y) = empirical_epsilon_laplacian_num_points(w1, w2, n, epsilon)

    fig, ax = plt.subplots()
    ax.set_xticks([0, pi/2, pi, 1.5*pi, 2*pi])
    ax.set_xlim([0, 2*pi])
    ax.set_xlabel(r'$\theta$')
    ax.set_ylabel(r'num points')
    ax.set_xticklabels([r'$0$', r'$\frac{\pi}{2}$', r'$\pi$', r'$\frac{3}{2} \pi$', r'$2 \pi$'])
    plt.plot(X, Y, '-', label='Empirical')
    ax.grid()

    line_segment_length = 2*epsilon/(w1*absolute(sin(X)) + w2*absolute(cos(X)))
    Ytheory = n*line_segment_length/(2*pi)
    plt.plot(X, Ytheory, '--', label='Theory')
    plt.legend()


def check_laplacian_empirically(w1, w2, n, sigma, n_eigvecs):
    Lhat = empirical_laplacian_gaussian_kernel(w1, w2, n, sigma)
    (evalshat, evecshat) = compute_eigvalsvecs(Lhat, n_eigvecs)
    for (i, eigvec) in enumerate(evecshat):
        evecshat[i] *= sign(eigvec[1])
    plot_unitcircle_eigvecs(evalshat, evecshat, None, 4)


def plot_complex_vector(v, name):
    fig, ax = plt.subplots()
    ax.set_xticks([0, pi/2, pi, 1.5*pi, 2*pi])
    ax.set_xlim([0, 2*pi])
    ax.set_xlabel(r'$\theta$')
    ax.set_ylabel(r'num points')
    ax.set_xticklabels([r'$0$', r'$\frac{\pi}{2}$', r'$\pi$', r'$\frac{3}{2} \pi$', r'$2 \pi$'])

    n = len(v)
    X = linspace(0,2*pi,n+1)[:n] 
    ax.plot(X, v.real, label=f'Re({name})')
    ax.plot(X, v.imag, label=f'Im({name})')
    ax.legend()


def example_function(X):
    (n,) = X.shape
    f = np.sin(X) + np.cos(2*X) + np.cos(5*X)
    return f.reshape((n,1))


def check_new_L1_laplacian_operator(n, sigma):
    W1 = 1
    W2 = 1.5
    (X_grid, L1_theoretical) = weighted_L1_laplacian_new(W1, W2, n)

    d = 1 # intrinsic dimension
    C_n = 2*pi / (gamma((d+4)/2.0) * sigma**(d+2))
    (X_sampled, L1_empirical_unscaled) = empirical_laplacian_gaussian_kernel(W1, W2, n, sigma, 5*sigma)
    L1_empirical = (C_n / n) * L1_empirical_unscaled

    rc_dbl_col = RCPARAMS_LATEX_DOUBLE_COLUMN
    with matplotlib.rc_context(rc = RCPARAMS_LATEX_DOUBLE_COLUMN):
        wide_figure_size = tuple([2,1] * np.array(RCPARAMS_LATEX_DOUBLE_COLUMN['figure.figsize']))
        (fig, ax) = plt.subplots(figsize = wide_figure_size)

        Y_theoretical = L1_theoretical * example_function(X_grid)
        ax.plot(X_grid, Y_theoretical, '--', label=r'$\Delta_{\mathcal{M}, \mathcal{B}} f$ (theory)')

        Y_empirical = L1_empirical * example_function(X_sampled)
        ax.plot(X_sampled, Y_empirical, '-', label=r'$\mathcal{L}_{n, \mathcal{B}} f$ (empirical)')

        ax.set_xlim([0, 2*pi])
        ax.set_xticks([0, pi/2, pi, 1.5*pi, 2*pi])
        ax.set_xticklabels([r'$0$', r'$\frac{\pi}{2}$', r'$\pi$', r'$\frac{3}{2} \pi$', r'$2 \pi$'])
        ax.legend(loc='upper center')
        dollarsign = r'$'
        ax.set_title(f'$n = {n}$')

        save_figure(fig, f'check_L1_laplacian_n={n}')


def plot_all_check_L1_laplacian():
    check_new_L1_laplacian_operator(4000, 0.1)
    check_new_L1_laplacian_operator(40000, 0.1)


def plot_all_first_eigenvectors():
    RESOLUTION = 100000
    N_EIGVECS = 5
    EIGENVALUE_LEGEND_N_DIGITS = 4

    plot_first_eigenvectors(1.001, 1, RESOLUTION, N_EIGVECS, EIGENVALUE_LEGEND_N_DIGITS)
    plot_first_eigenvectors(2, 1, RESOLUTION, N_EIGVECS, EIGENVALUE_LEGEND_N_DIGITS)
    plot_first_eigenvectors(4, 1, RESOLUTION, N_EIGVECS, EIGENVALUE_LEGEND_N_DIGITS)
    plot_first_eigenvectors(8, 1, RESOLUTION, N_EIGVECS, EIGENVALUE_LEGEND_N_DIGITS)

