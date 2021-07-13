
# Settings to make nice LaTeX-compatible figures.
#
# Use as follows:
#
# with matplotlib.rc_context(rc = RCPARAMS_LATEX_DOUBLE_COLUMN):
#     fig, ax = plt.subplots()
#     ax.plot(...)
#     ...
#     save_figure(fig, 'very-plot-such-amazing-wow')

import matplotlib

FIGURES_PATH = 'figures/'
DPI = 600


_RCPARAMS_LATEX_SINGLE_COLUMN = {
    'font.family': 'serif',
    'text.usetex': True,

    'axes.labelsize': 13,
    'axes.titlesize': 12,
    'legend.fontsize': 9,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
     
    'axes.prop_cycle': matplotlib.pyplot.cycler('color', ['#006496', '#ff816b', '#fbca60', '#6d904f', '#8b8b8b']) + matplotlib.pyplot.cycler('marker', ['o', 'd', 's', '*', '>']),

    'lines.markersize': 4,
    'lines.markeredgewidth': 0.5,
    'lines.markeredgecolor': 'k',
                               
    'legend.fancybox': True, # Rounded legend box
    'legend.framealpha': 0.8,

    'axes.linewidth': 0.5,
}

# This is the right width (in inches) for a 'letter' page LaTeX document that imports the geometry package with default parameters.
_PAGE_WIDTH_INCHES = 6.775
_GOLDEN_RATIO = (5**0.5 - 1)/2
RCPARAMS_LATEX_DOUBLE_COLUMN = {**_RCPARAMS_LATEX_SINGLE_COLUMN, 'figure.figsize': (_PAGE_WIDTH_INCHES/2, _GOLDEN_RATIO*_PAGE_WIDTH_INCHES/2)}


def save_figure(fig, name):
    import os
    os.makedirs(FIGURES_PATH, exist_ok=True)
    filename = os.path.join(FIGURES_PATH, name).replace('.','_') + '.pdf'
    print('Saving figure to', filename)
    fig.savefig(filename, dpi=DPI, bbox_inches='tight')

