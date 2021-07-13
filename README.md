# Manifold Learning with Arbitrary Norms

Requirements:
* Python 3 with NumPy, SciPy, Matplotlib. I recommend installing the Anaconda Python distribution.
* mrcfile package


This code produces the figures of the paper (under review):
    "Manifold learning with arbitrary norms" by Joe Kileel, Amit Moscovich, Nathan Zelesko, Amit Singer.
    https://arxiv.org/abs/2012.14172
    
The above paper is an extension of the following short conference paper by the same authors, which also has some of the figures:
    "Earthmover-based manifold learning for analyzing molecular conformation spaces"
    IEEE 17th International Symposium on Biomedical Imaging (ISBI), 2020.
    Publisher version: https://ieeexplore.ieee.org/document/9098723
    arXiv version: https://arxiv.org/abs/1911.06107


Running `produce_all_figures.py` performs all the computations and produces all figures into the figures/ directory.
It takes a few minutes.


# Prerequisites

Python 3 is required with the following packages:
* NumPy
* SciPy
* scikit-learn
* mrcfile

The easiest way to install these is to download the Anaconda Python distribution and then run "pip install mrcfile".

Since the figures use latex rendering for the labels, you need:
* TeXLive. The `latex` binary must be in the command path.
* dvipdf and dvipng
(or you can just remove the TeX code from the labels used in plotting the figures)


# Contact

If you're interested in this code or associated research, feel free to shoot me an email.

Amit Moscovich
amit@moscovich.org
