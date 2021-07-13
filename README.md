# Manifold Learning with Arbitrary Norms

This code reproduces the figures from the paper:
    "Manifold learning with arbitrary norms" by Joe Kileel, Amit Moscovich, Nathan Zelesko, Amit Singer.
   https://arxiv.org/abs/2012.14172 (arXiv)
    
The numerical section of the above paper extends the following conference paper by the same authors:
    "Earthmover-based manifold learning for analyzing molecular conformation spaces"
    IEEE 17th International Symposium on Biomedical Imaging (ISBI), 2020.
    https://ieeexplore.ieee.org/document/9098723 (published), https://arxiv.org/abs/1911.06107 (arXiv)


# Prerequisites

Python 3 is required with the following packages:
* NumPy
* SciPy
* scikit-learn
* mrcfile

The easiest way to install these is to download the Anaconda Python distribution and then run "pip install mrcfile".

Since the figures use latex rendering for the labels, you need:
* TeXLive (the `latex` binary must be in the command path);
* dvipdf and dvipng.
Or you can just remove the TeX code from the labels used in plotting the figures.


# How to run

Running `produce_all_figures.py` performs all the computations and produces all figures into the figures/ directory.
It takes a few minutes.


# Contact

If you have any questions, feel free to email:
amit@moscovich.org
