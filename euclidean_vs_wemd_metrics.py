import os

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
import matplotlib.pyplot as plt
from numpy.random import normal

import ATP_synthase_datagen
import wemd
import euclidean_vs_wemd_embeddings


def calculate_wemds(data, noise_std, wavelet, level):
    shape = data[0].shape
    wemds = []
    for i in range(len(data)):
        embedding_0 = wemd.embed(data[0]+normal(0, noise_std, shape), wavelet, level)
        embedding_i = wemd.embed(data[i]+normal(0, noise_std, shape), wavelet, level)
        wemds.append(norm(embedding_0-embedding_i, ord=1))

    return wemds


def calculate_euclid_dists(data, noise_std):
    stdev_2 = (2**0.5)*noise_std
    shape = data[0].shape
    
    return [norm(data[0]-data[i]+normal(0, stdev_2, shape)) for i in range(len(data))]
    
    
def emd_vs_euclid_plot(data, noise_std, angles, wavelet, level, filename):
    for i in range(len(angles)):
        if angles[i] > 180:
            angles[i] = -(360 - angles[i])
    print('Calculating WEMD distances')
    emds = calculate_wemds(data, noise_std, wavelet, level)
    print('Calculating Euclidean distances')
    eucs = calculate_euclid_dists(data, noise_std)
    
    sorted_emds = np.array([x for y, x in sorted(zip(angles,emds))])
    sorted_eucs = np.array([x for y, x in sorted(zip(angles,eucs))])
    angles.sort()
    
    sorted_emds = sorted_emds - min(sorted_emds)
    sorted_eucs = sorted_eucs - min(sorted_eucs)
    maxemd = max(sorted_emds)
    maxeuc = max(sorted_eucs)
    s = maxemd/maxeuc
    sorted_eucs = s*sorted_eucs
    
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["axes.linewidth"] = 0.5 
    plt.rcParams['xtick.minor.width'] = 0.5
    plt.rcParams['xtick.major.width'] = 0.5
    plt.rcParams['legend.fontsize'] = 8
    plt.rcParams['text.usetex'] = True
    plt.rcParams['mathtext.fontset'] = 'cm' # Computer-modern font that is used in LaTeX
    plt.rcParams['pdf.fonttype'] = 42 # This gets rid of "Type 3 font" errors in the IEEE compliance system

    fig,ax = plt.subplots(figsize=(5.75, 2.1631))
    plt.plot(angles,sorted_eucs,'--',label="Euclidean")
    plt.plot(angles,sorted_emds,label="WEMD")
    plt.legend()
    plt.xticks([-180,-150,-120,-90,-60,-30,0,30,60,90,120,150,180],['-180$^{\circ}$','-150$^{\circ}$','-120$^{\circ}$','-90$^{\circ}$','-60$^{\circ}$','-30$^{\circ}$','0$^{\circ}$','30$^{\circ}$','60$^{\circ}$','90$^{\circ}$','120$^{\circ}$','150$^{\circ}$','180$^{\circ}$'])
    plt.yticks([])
    plt.legend()
    ax.tick_params(labelsize=7)
    plt.xlim(-180,180)
    plt.savefig(filename,pad_inches=0, bbox_inches='tight')  
    
    
def compute_and_plot(wavelet, level, figure_dir):
    EQUIDISTANT_ANGLES = np.linspace(0, 360, 361)[:360]
    (data, angles) = ATP_synthase_datagen.build_dataset(360, EQUIDISTANT_ANGLES)

    emd_vs_euclid_plot(data, 0, angles.copy(), wavelet, level, os.path.join(figure_dir,f'L2vsWEMD_noiseless.pdf'))

    STD = euclidean_vs_wemd_embeddings.STD
    emd_vs_euclid_plot(data, STD, angles.copy(), wavelet, level, os.path.join(figure_dir,f'L2vsWEMD_noisy.pdf'))

