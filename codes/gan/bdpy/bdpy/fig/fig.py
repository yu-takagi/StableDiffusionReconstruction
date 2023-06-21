'''Figure module

This file is a part of BdPy.

Functions
---------
makefigure
    Create a figure
box_off
    Remove upper and right axes
draw_footnote
    Draw footnote on a figure
'''


__all__ = [
    'box_off',
    'draw_footnote',
    'make_violinplots',
    'makefigure',
]


import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def makefigure(figtype='a4landscape'):
    '''Create a figure'''

    if figtype == 'a4landscape':
        figsize = (11.7, 8.3)
    elif figtype == 'a4portrait':
        figsize = (8.3, 11.7)
    else:
        raise ValueError('Unknown figure type %s' % figtype)

    return plt.figure(figsize=figsize)


def box_off(ax):
    '''Remove upper and right axes'''

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')


def draw_footnote(fig, string, fontsize=9):
    '''Draw footnote on a figure'''
    ax = fig.add_axes([0., 0., 1., 1.])
    ax.text(0.5, 0.01, string, horizontalalignment='center', fontsize=fontsize)
    ax.patch.set_alpha(0.0)
    ax.set_axis_off()

    return ax


def make_violinplots(df, x=None, y=None, subplot=None, figure=None, x_list=None, subplot_list=None, figure_list=None, title=None, x_label=None, y_label=None, fontsize=16, points=100):

    x_keys = sorted(df[x].unique())
    subplot_keys = sorted(df[subplot].unique())
    figure_keys = sorted(df[figure].unique())

    x_list = x_keys if x_list is None else x_list
    subplot_list = subplot_keys if subplot_list is None else subplot_list
    figure_list = figure_keys if figure_list is None else figure_list

    print('X:       {}'.format(x_list))
    print('Subplot: {}'.format(subplot_list))
    print('Figures: {}'.format(figure_list))

    col_num = np.ceil(np.sqrt(len(subplot_list)))
    row_num = int(np.ceil(len(subplot_list) / col_num))
    col_num = int(col_num)

    print('Subplot in {} x {}'.format(row_num, col_num))

    figs = []

    # Figure loop
    for fig_label in figure_list:
        print('Creating figure for {}'.format(fig_label))
        fig = makefigure('a4landscape')

        sns.set()
        sns.set_style('ticks')
        sns.set_palette('gray')

        # Subplot loop
        for i, sp_label in enumerate(subplot_list):
            print('Creating subplot for {}'.format(sp_label))

            # Set subplot position
            col = int(i / row_num)
            row = i - col * row_num
            sbpos = (row_num - row - 1) * col_num + col + 1

            # Get data
            data = []
            for j, x_lbl in enumerate(x_list):
                df_t = df.query('{} == "{}" & {} == "{}" & {} == "{}"'.format(subplot, sp_label, figure, fig_label, x, x_lbl))
                data_t = df_t[y].values
                data_t = np.array([np.nan, np.nan]) if len(data_t) == 0 else np.concatenate(data_t)
                # violinplot requires at least two elements in the dataset
                data.append(data_t)

            # Plot
            ax = plt.subplot(row_num, col_num, sbpos)

            ax.hlines(0, xmin=-1, xmax=len(x_list), color='k', linestyle='-', linewidth=0.5)
            ax.hlines([-0.4, -0.2, 0.2, 0.4, 0.6, 0.8], xmin=-1, xmax=len(x_list), color='k', linestyle=':', linewidth=0.5)

            xpos = range(len(x_list))

            ax.violinplot(data, xpos, showmeans=True, showextrema=False, showmedians=False, points=points)

            ax.text(-0.5, 0.85, sp_label, horizontalalignment='left', fontsize=fontsize)

            ax.set_xlim([-1, len(x_list)])
            ax.set_xticks(range(len(x_list)))
            if row == 0:
                ax.set_xticklabels(x_list, rotation=-45, fontsize=fontsize)
            else:
                ax.set_xticklabels([])

            ax.set_ylim([-0.4, 1.0])  # FXIME: auto-scaling
            ax.tick_params(axis='y', labelsize=fontsize)
            box_off(ax)

            plt.tight_layout()

        # X Label
        if x_label is not None:
            ax = fig.add_axes([0, 0, 1, 1])
            ax.text(0.5, 0, x_label,
                    verticalalignment='center', horizontalalignment='center', fontsize=fontsize)
            ax.patch.set_alpha(0.0)
            ax.set_axis_off()

        # Y label
        if y_label is not None:
            ax = fig.add_axes([0, 0, 1, 1])
            ax.text(0, 0.5, y_label,
                    verticalalignment='center', horizontalalignment='center', fontsize=fontsize, rotation=90)
            ax.patch.set_alpha(0.0)
            ax.set_axis_off()

        # Figure title
        if title is not None:
            ax = fig.add_axes([0, 0, 1, 1])
            ax.text(0.5, 0.99, '{}: {}'.format(title, fig_label),
                    horizontalalignment='center', fontsize=fontsize)
            ax.patch.set_alpha(0.0)
            ax.set_axis_off()

        figs.append(fig)

    if len(figs) == 1:
        return figs[0]
    else:
        return figs
