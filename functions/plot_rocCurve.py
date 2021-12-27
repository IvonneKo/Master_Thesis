import numbers

import six

import numpy
import matplotlib.collections
from matplotlib import pyplot
import matplotlib.pyplot as plt
# using example from
# http://nbviewer.ipython.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb


def make_segments(x, y):
    '''
    Create list of line segments from x and y coordinates,
    in the correct format for LineCollection:
    an array of the form
    numlines x (points per line) x 2 (x and y) array
    '''

    points = numpy.array([x, y]).T.reshape(-1, 1, 2)
    segments = numpy.concatenate([points[:-1], points[1:]], axis=1)

    return segments


def colorline(x, y, z=None, axes=None,
              cmap=pyplot.get_cmap('Spectral_r'),
              norm=pyplot.Normalize(0.0, 0.1), linewidth=5, alpha=0.8,
              **kwargs):
    '''
    Plot a colored line with coordinates x and y
    Optionally specify colors in the array z
    Optionally specify a colormap, a norm function and a line width
    '''

    # Default colors equally spaced on [0,1]:
    if z is None:
        z = numpy.linspace(0.0, 1.0, len(x))

    # Special case if a single number:
    if isinstance(z, numbers.Real):
        z = numpy.array([z])

    z = numpy.asarray(z)

    segments = make_segments(x, y)
    lc = matplotlib.collections.LineCollection(
        segments, array=z, cmap=cmap, norm=norm,
        linewidth=linewidth, alpha=alpha, **kwargs
    )

    if axes is None:
        axes = pyplot.gca()

    axes.add_collection(lc)
    axes.autoscale()

    return lc


def plot_roc(tpr, fpr, thresholds, subplots_kwargs=None,
             label_every=None, label_kwargs=None,
             fpr_label='1-Specificity',
             tpr_label='Sensitivity',
             luck_label='Random',
             title='Receiver operating characteristic',
             **kwargs):
    
   

    if subplots_kwargs is None:
        subplots_kwargs = {}

    figure, axes = pyplot.subplots(1, 1, **subplots_kwargs)

    if 'lw' not in kwargs:
        kwargs['lw'] = 1

    axes.plot(fpr, tpr, **kwargs)

    if label_every is not None:
        if label_kwargs is None:
            label_kwargs = {}

        if 'bbox' not in label_kwargs:
            label_kwargs['bbox'] = dict(
                boxstyle='round,pad=0.5', fc='white', alpha=0.5,
            )

        for k in six.moves.range(len(tpr)):
            if k % label_every != 0:
                continue

            threshold = str(numpy.round(thresholds[k], 3))
            x = fpr[k]
            y = tpr[k]
            axes.annotate(threshold, (x, y), **label_kwargs,  fontsize=10)

    if luck_label is not None:
        axes.plot((0, 1), (0, 1), '--', color='Gray', label=luck_label)
       

    lc = colorline(fpr, tpr, thresholds, axes=axes)
    figure.colorbar(lc)
    
    plt.rcParams["figure.figsize"] = (10,8)
   # plt.rcParams["figure.figsize"] = plt.rcParamsDefault["figure.figsize"]
    axes.set_xlim([-0.05, 1.05])
    axes.set_ylim([-0.05, 1.05])

    axes.set_xlabel(fpr_label, fontsize=20)
    axes.set_ylabel(tpr_label,fontsize=20)

    axes.set_title(title, fontsize=20)

    axes.legend(loc="lower right")
    plt.xticks(size = 20)
    plt.yticks(size = 20)
    


    return figure, axes