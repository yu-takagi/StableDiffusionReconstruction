import warnings

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import numpy as np
import pandas as pd
import seaborn as sns

from bdpy.fig import box_off


def makeplots(
        df,
        x=None, y=None,
        x_list=None,
        subplot=None, subplot_list=None,
        figure=None, figure_list=None,
        group=None, group_list=None,
        bar_group_width=0.8,
        plot_type='bar', horizontal=False, ebar=None,
        plot_size_auto=True, plot_size=(4, 0.3),
        max_col=None,
        y_lim=None, y_ticks=None,
        title=None, x_label=None, y_label=None,
        fontsize=12, tick_fontsize=9, points=100,
        style='default', colorset=None,
        chance_level=None, chance_level_style={'color': 'k', 'linewidth': 1},
        swarm_dot_color='gray',
        swarm_dot_size=3, swarm_dot_alpha=0.7,
        swarm_violin_color='blue',
        box_color='blue', box_width=0.5, box_linewidth=1,
        box_meanprops=dict(linestyle='-', linewidth=1.5, color='red'),
        box_medianprops={},
        removenan=True,
        verbose=False, colors=None, reverse_x=False
):
    '''Make plots.
    Parameters
    ----------
    df : pandas.core.frame.DataFrame
    x : str
    y : str
    x_list : list
    subplot : str
    subplot : list
    figure : str
    figure_list : list
    plot_type : {'bar', 'violin', 'paired violin', 'swarm', 'swarm+box'}
    horizontal: bool
    plot_size : (width, height)
    y_lim : (y_min, y_max)
    y_ticks : array_like
    title, x_label, y_label : str
    fontsize : int
    tick_fontsize : int
    style : str
    verbose : bool
    Returns
    -------
    fig : matplotlib.figure.Figure or list of matplotlib.figure.Figure
    '''

    x_keys       = sorted(df[x].unique())
    subplot_keys = sorted(df[subplot].unique()) if subplot is not None else [None]
    figure_keys  = sorted(df[figure].unique()) if figure is not None else [None]
    group_keys   = sorted(df[group].unique()) if group is not None else [None]

    x_list       = x_keys       if x_list       is None else x_list
    subplot_list = subplot_keys if subplot_list is None else subplot_list
    figure_list  = figure_keys  if figure_list  is None else figure_list
    group_list   = group_keys   if group_list   is None else group_list

    if reverse_x:
        x_list = x_list[::-1]

    grouping = group is not None

    if plot_type == 'paired violin':
        if not grouping:
            RuntimeError('plot type "paired violin" can be used only when grouping is enabled')
        comparison_pairs = list(__split_list(group_list, 2))

    if grouping:
        warnings.warn('"grouping mode" is still experimental and will not work correctly yet!')

    if verbose:
        print('X:       {}'.format(x_list))
        if grouping:
            print('Group by: {} ({})'.format(group_keys, group_list))
        if subplot is not None:
            print('Subplot: {}'.format(subplot_list))
        if figure is not None:
            print('Figures: {}'.format(figure_list))

    col_num = np.ceil(np.sqrt(len(subplot_list)))
    row_num = int(np.ceil(len(subplot_list) / col_num))
    col_num = int(col_num)

    if max_col is not None and col_num > max_col:
        col_num = max_col
        row_num = int(np.ceil(len(subplot_list) / col_num))

    # Plot size
    if plot_size_auto:
        if horizontal:
            plot_size = (plot_size[0], plot_size[1] * len(x_list))
        else:
            plot_size = (plot_size[0] * len(x_list), plot_size[1])

    # Figure size
    figsize = (col_num * plot_size[0], row_num * plot_size[1])  # (width, height)

    if verbose:
        print('Subplot in {} x {}'.format(row_num, col_num))

    # Figure instances
    if plot_type == 'paired violin':
        figure_instances = [
            {
                'label': f,
                'comparison pair': p
             }
            for f in figure_list
            for p in comparison_pairs
        ]
    else:
        figure_instances = [
            {
                'label': f
             }
            for f in figure_list
        ]

    figs = []

    # Figure loop
    for figure_instance in figure_instances:
        fig_label = figure_instance['label']
        if verbose:
            if fig_label is None:
                print('Creating a figure')
            else:
                print('Creating figure for {}'.format(fig_label))

        plt.style.use(style)

        fig = plt.figure(figsize=figsize)

        # Subplot loop
        for i, sp_label in enumerate(subplot_list):
            if verbose:
                print('Creating subplot for {}'.format(sp_label))

            # Set subplot position
            col = int(i / row_num)
            row = i - col * row_num
            sbpos = (row_num - row - 1) * col_num + col + 1

            # Get data
            if plot_type == 'paired violin':
                group_list = figure_instance['comparison pair']

            if plot_type == "swarm+box":
                df_t = __strict_data(
                    df, subplot, sp_label,
                    figure, fig_label, y, removenan
                )
                weird_keys = []
                for key_candidate in [figure, subplot, group, x]:
                    if key_candidate is not None:
                        weird_keys.append(key_candidate)
                df_t = __weird_form_to_long(df_t, y, identify_cols = weird_keys)

            else:
                data = __get_data(
                    df, subplot, sp_label,
                    x, x_list, figure, fig_label, y,
                    group, group_list, grouping, removenan
                )

                if not isinstance(sp_label, list):
                    if grouping:
                        data_mean = [[np.nanmean(d) for d in data_t] for data_t in data]
                    else:
                        data_mean = [np.nanmean(d) for d in data]
                else:
                    data_mean = None

            # Plot
            ax = plt.subplot(row_num, col_num, sbpos)

            if not style == 'ggplot':
                if horizontal:
                    ax.grid(axis='x', color='k', linestyle='-', linewidth=0.5)
                else:
                    ax.grid(axis='y', color='k', linestyle='-', linewidth=0.5)

            xpos = range(len(x_list))

            if plot_type == 'bar':
                __plot_bar(
                    ax, xpos, data_mean,
                    horizontal=horizontal,
                    grouping=grouping, group_list=group_list,
                    bar_group_width=bar_group_width
                )
            elif plot_type == 'violin':
                group_label_list = __plot_violin(
                    ax, xpos, data,
                    horizontal=horizontal,
                    grouping=grouping, group_list=group_list,
                    bar_group_width=bar_group_width, points=points
                )
            elif plot_type == 'paired violin':
                group_label_list = __plot_violin_paired(
                    ax, xpos, data,
                    horizontal=horizontal,
                    grouping=grouping, group_list=group_list,
                    points=points, colors=colors
                )
            elif plot_type == 'swarm':
                __plot_swarm(
                    ax, x_list, data,
                    horizontal=horizontal, grouping=grouping,
                    dot_color=swarm_dot_color,
                    dot_size=swarm_dot_size,
                    dot_alpha=swarm_dot_alpha,
                    violin_color=swarm_violin_color,
                )
            elif plot_type == 'swarm+box':
                legend_handler = __plot_swarmbox(
                    ax, x, y, x_list, df_t,
                    horizontal=horizontal, reverse_x=reverse_x,
                    grouping=grouping, group=group, group_list=group_list,
                    dot_color=swarm_dot_color,
                    dot_size=swarm_dot_size,
                    dot_alpha=swarm_dot_alpha,
                    box_color=box_color, box_width=box_width,
                    box_linewidth=box_linewidth,
                    box_meanprops=box_meanprops,
                    box_medianprops=box_medianprops
                )
            else:
                raise ValueError('Unknown plot_type: {}'.format(plot_type))

            if not horizontal:
                # Vertical plot
                ax.set_xlim([-1, len(x_list)])
                ax.set_xticks(range(len(x_list)))

                if row == 0:
                    ax.set_xticklabels(x_list, rotation=-45, ha='left', fontsize=tick_fontsize)
                else:
                    ax.set_xticklabels([])

                if y_lim is None:
                    pass
                else:
                    ax.set_ylim(y_lim)

                if y_ticks is not None:
                    ax.set_yticks(y_ticks)

                ax.tick_params(axis='y', labelsize=tick_fontsize, grid_color='gray', grid_linestyle='--', grid_linewidth=0.8)

                if chance_level is not None:
                    plt.hlines(chance_level, xmin=plt.gca().get_xlim()[0], xmax=plt.gca().get_xlim()[1], **chance_level_style)
            else:
                # Horizontal plot
                ax.set_ylim([-1, len(x_list)])
                ax.set_yticks(range(len(x_list)))

                if col == 0:
                    ax.set_yticklabels(x_list, fontsize=tick_fontsize)
                else:
                    ax.set_yticklabels([])

                if y_lim is None:
                    pass
                else:
                    ax.set_xlim(y_lim)

                if y_ticks is not None:
                    ax.set_xticks(y_ticks)

                ax.tick_params(axis='x', labelsize=tick_fontsize, grid_color='gray', grid_linestyle='--', grid_linewidth=0.8)

                if chance_level is not None:
                    plt.vlines(chance_level, ymin=plt.gca().get_ylim()[0], ymax=plt.gca().get_ylim()[1], **chance_level_style)

            # Inset title
            x_range = plt.gca().get_xlim()[1] - plt.gca().get_xlim()[0]
            y_range = plt.gca().get_ylim()[1] - plt.gca().get_ylim()[0]
            tpos = (
                plt.gca().get_xlim()[0] + 0.03 * x_range,
                plt.gca().get_ylim()[1] + 0.01 * y_range
            )

            ax.text(tpos[0], tpos[1], sp_label, horizontalalignment='left', verticalalignment='top', fontsize=fontsize, bbox=dict(facecolor='white', edgecolor='none'))

            # Inset legend
            if grouping:
                if 'violin' in plot_type:
                    if i == len(subplot_list) - 1:
                        group_label_list = group_label_list[::-1]
                        ax.legend(*zip(*group_label_list), loc='upper left', bbox_to_anchor=(1, 1))
                elif plot_type == 'swarm+box':
                    pass
                else:
                    plt.legend()

            box_off(ax)

            plt.tight_layout()

        # Draw legend, X/Y labels, and title ----------------------------

        # Legend
        if plot_type == 'swarm+box':
            if legend_handler is not None:
                if len(subplot_list) < col_num*row_num:
                    ax = plt.subplot(row_num, col_num, col_num)
                else:
                    ax = fig.add_axes([1, 0.5, 1./col_num*0.6, 0.5])
                ax.legend(legend_handler[0], legend_handler[1],
                          loc='upper left', bbox_to_anchor=(0, 1.0), fontsize=tick_fontsize)
                ax.set_axis_off()
                plt.tight_layout()

        ax = fig.add_axes([0, 0, 1, 1])
        ax.patch.set_alpha(0.0)
        ax.set_axis_off()

        # X Label
        if x_label is not None:
            txt = y_label if horizontal else x_label
            ax.text(0.5, 0, txt, verticalalignment='center', horizontalalignment='center', fontsize=fontsize)

        # Y label
        if y_label is not None:
            txt = x_label if horizontal else y_label
            ax.text(0, 0.5, txt, verticalalignment='center', horizontalalignment='center', fontsize=fontsize, rotation=90)

        # Figure title
        if title is not None:
            if fig_label is None:
                ax.text(0.5, 0.99, title, horizontalalignment='center', fontsize=fontsize)
            else:
                ax.text(0.5, 0.99, '{}: {}'.format(title, fig_label), horizontalalignment='center', fontsize=fontsize)

        figs.append(fig)

    if figure is None:
        return figs[0]
    else:
        return figs


def __plot_bar(
        ax, xpos, data_mean,
        horizontal=False,
        grouping=False, group_list=[],
        bar_group_width=0.8
):
    if grouping:
        ydata = np.array(data_mean)
        n_grp = ydata.shape[1]
        w = bar_group_width / n_grp

        for grpi in range(n_grp):
            offset = grpi * w
            if horizontal:
                plt.barh(np.array(xpos) - bar_group_width / 2 + (bar_group_width / 2) * w + offset, ydata[:, grpi], height=w, label=group_list[grpi])
            else:
                plt.bar(np.array(xpos) - bar_group_width / 2 + (bar_group_width / 2) * w + offset, ydata[:, grpi], width=w, label=group_list[grpi])
    else:
        if horizontal:
            ax.barh(xpos, data_mean, color='gray')
        else:
            ax.bar(xpos, data_mean, color='gray')


def __plot_violin(
        ax, xpos, data,
        horizontal=False,
        grouping=False, group_list=[],
        bar_group_width=0.8, points=100
):
    if grouping:
        n_grp = len(group_list)
        w = bar_group_width / (n_grp + 1)

        group_label_list = []
        for grpi in range(n_grp):
            offset = grpi * w - (n_grp // 2) * w
            xpos_grp = np.array(xpos) + offset  #- bar_group_width / 2 + (bar_group_width / 2) * w + offset
            ydata_grp = [a_data[grpi] for a_data in data]
            violinobj = ax.violinplot(
                ydata_grp, xpos_grp,
                vert=not horizontal,
                showmeans=True, showextrema=False, showmedians=False, points=points,
                widths=w * 0.8)
            color = violinobj["bodies"][0].get_facecolor().flatten()
            group_label_list.append((mpatches.Patch(color=color), group_list[grpi]))
    else:
        ax.violinplot(data, xpos, vert=not horizontal, showmeans=True, showextrema=False, showmedians=False, points=points)
        group_label_list = None

    return group_label_list


def __plot_violin_paired(
        ax, xpos, data,
        horizontal=False,
        grouping=False, group_list=[],
        points=100, colors=None
):
    assert grouping
    n_grp = len(group_list)
    assert n_grp == 2

    group_label_list = []
    if colors is not None and len(colors) >= 2:
        __draw_half_violin(ax, [a_data[0] for a_data in data], points, xpos, color=colors[0], left=True, vert=not horizontal)
        __draw_half_violin(ax, [a_data[1] for a_data in data], points, xpos, color=colors[1], left=False, vert=not horizontal)
    else:
        colors = []
        color = __draw_half_violin(ax, [a_data[0] for a_data in data], points, xpos, color=None, left=True, vert=not horizontal)
        colors.append(color)
        color = __draw_half_violin(ax, [a_data[1] for a_data in data], points, xpos, color=None, left=False, vert=not horizontal)
        colors.append(color)
    for color, label in zip(colors, group_list):
        group_label_list.append((mpatches.Patch(color=color), label))

    return group_label_list


def __plot_swarm(
        ax, x_list, data,
        horizontal=False,
        grouping=False,
        dot_color='#595959', dot_size=1.5, dot_alpha=0.8,
        violin_color='blue'
):
    if grouping:
        raise RuntimeError("The function of grouping on `swarm` plot is not implemeted yet.")
    else:
        df_list = []
        for xi, x_lbl in enumerate(x_list):
            a_df = pd.DataFrame.from_dict({'y': data[xi]})
            a_df['x'] = x_lbl
            df_list.append(a_df)
        tmp_df = pd.concat(df_list)
        mean_df = tmp_df.groupby('x', as_index=False).mean()
        mean_list = [mean_df[mean_df['x'] == x_lbl]['y'].values[0] for x_lbl in x_list]
        if horizontal:
            plotx, ploty = 'y', 'x'
            scatterx, scattery = mean_list, np.arange(len(x_list))
            scattermark = "|"
        else:
            plotx, ploty = 'x', 'y'
            scatterx, scattery = np.arange(len(x_list)), mean_list
            scattermark = "_"
        ax = sns.violinplot(
            x=plotx, y=ploty, order=x_list, orient="h" if horizontal else "v",
            data=tmp_df, ax=ax, color=violin_color, linewidth=0
        )
        for violin in ax.collections[::2]:
            violin.set_alpha(0.6)
        sns.swarmplot(
            x=plotx, y=ploty, order=x_list, orient="h" if horizontal else "v",
            data=tmp_df, ax=ax, color=dot_color, alpha=dot_alpha, size=dot_size
        )
        ax.scatter(x=scatterx, y=scattery, marker=scattermark, c="red", linewidths=2, zorder=10)

        ax.set(xlabel=None, ylabel=None)


def __plot_swarmbox(
        ax, x, y, x_list, df_t,
        horizontal=False, reverse_x=False,
        grouping=False, group=None, group_list=[],
        dot_color='#696969', dot_size=3, dot_alpha=0.7,
        box_color='blue', box_width=0.5, box_linewidth=1, box_props={'alpha': .3},
        box_meanprops=dict(linestyle='-', linewidth=1.5, color='red'),
        box_medianprops={}
):
    # warning
    if grouping:
        warnings.warn('When grouping is True, "box_width" is not working to make the layout consistent with the swarm plot.')

    # prepare plot
    box_color_palette = sns.color_palette("pastel") if grouping else None
    dot_color_palette = sns.color_palette("bright") if grouping else None
    if horizontal:
        plotx, ploty = y, x
    else:
        plotx, ploty = x, y

    # plot swarmplot
    if grouping:
        sns.swarmplot(
            data=df_t, ax=ax,
            x=plotx, y=ploty, order=x_list, hue=group, hue_order=group_list,
            orient="h" if horizontal else "v",
            palette=dot_color_palette,
            dodge=True,
            color=dot_color, size=dot_size, alpha=dot_alpha, zorder=10
        )
    else:
        sns.swarmplot(
            data=df_t, ax=ax,
            x=plotx, y=ploty, order=x_list,
            orient="h" if horizontal else "v",
            color=dot_color, size=dot_size, alpha=dot_alpha, zorder=10
        )

    # plot boxplot
    if grouping:
        boxax = sns.boxplot(
            data=df_t, ax=ax,
            x=plotx, y=ploty, order=x_list, hue=group, hue_order=group_list,
            orient="h" if horizontal else "v",
            palette=dot_color_palette,
            linewidth=box_linewidth,
            showfliers=False,
            showmeans=True, meanline=True, meanprops=box_meanprops,
            medianprops=box_medianprops,
            boxprops=box_props, zorder=100  # <- This zorder is very important for visualization.
        )
        # prepare legend
        handlers, labels = boxax.get_legend_handles_labels()
        handlers = handlers[:len(group_list)]
        labels = labels[:len(group_list)]
        if horizontal:
            legend_handler = [handlers[::-1], labels[::-1]]
        else:
            legend_handler = [handlers, labels]
        ax.get_legend().remove()
    else:
        sns.boxplot(
            data=df_t, ax=ax,
            x=plotx, y=ploty, order=x_list,
            orient="h" if horizontal else "v",
            color=box_color,
            linewidth=box_linewidth,
            width=box_width,
            showfliers=False,
            showmeans=True, meanline=True, meanprops=box_meanprops,
            medianprops=box_medianprops,
            boxprops=box_props, zorder=100  # <- This zorder is very important for visualization.
        )
        legend_handler = None

    # remove label
    ax.set(xlabel=None, ylabel=None)

    return legend_handler


def __split_list(l, n):
    for idx in range(0, len(l), n):
        yield l[idx:idx + n]


def __get_data(
        df, subplot, sp_label,
        x, x_list, figure, fig_label, y,
        group, group_list, grouping, removenan
):
    data = []
    for j, x_lbl in enumerate(x_list):
        if grouping:
            data_t = []
            for group_label in group_list:
                if fig_label is None:
                    df_t = df.query('`{}` == "{}" & `{}` == "{}" & `{}` == "{}"'.format(subplot, sp_label, group, group_label, x, x_lbl))
                else:
                    df_t = df.query('`{}` == "{}" & `{}` == "{}" & `{}` == "{}" & `{}` == "{}"'.format(subplot, sp_label, group, group_label, figure, fig_label, x, x_lbl))
                data_tt = df_t[y].values
                if removenan:
                    data_tt[0] = np.delete(data_tt[0], np.isnan(data_tt[0]))  # FXIME
                data_tt = np.array([np.nan, np.nan]) if len(data_tt) == 0 else np.concatenate(data_tt)
                data_t.append(data_tt)
            # violinplot requires at least two elements in the dataset
        else:
            if fig_label is None:
                df_t = df.query('`{}` == "{}" & `{}` == "{}"'.format(subplot, sp_label, x, x_lbl))
            else:
                df_t = df.query('`{}` == "{}" & `{}` == "{}" & `{}` == "{}"'.format(subplot, sp_label, figure, fig_label, x, x_lbl))
            data_t = df_t[y].values
            if removenan:
                data_t[0] = np.delete(data_t[0], np.isnan(data_t[0]))  # FXIME
            data_t = np.array([np.nan, np.nan]) if len(data_t) == 0 else np.concatenate(data_t)
            # violinplot requires at least two elements in the dataset

        data.append(data_t)

    return data


def __strict_data(
        df, subplot, sp_label, figure, fig_label, y, removenan
):
    if fig_label is None and sp_label is None:
        df_t = df
    elif fig_label is None:
        df_t = df.query('`{}` == "{}"'.format(subplot, sp_label))
    else:
        df_t = df.query('`{}` == "{}" & `{}` == "{}"'.format(subplot, sp_label, figure, fig_label))
    df_t = df_t.reset_index(drop=True)

    if removenan:
        df_t[y] = df_t[y].apply(lambda x: np.delete(x, np.isnan(x)))

    return df_t


def __weird_form_to_long(df, target_col, identify_cols=[]):
    df_result = pd.DataFrame()
    for i, row in df.iterrows():
        tmp = {}
        for col in identify_cols:
            tmp[col] = row[col]
        tmp[target_col] = row[target_col]
        df_result = pd.concat([df_result, pd.DataFrame(tmp)])
    return df_result


def __draw_half_violin(
        ax, data, points, positions,
        color=None, left=True, vert=True
):
    v = ax.violinplot(data, points=points, positions=positions, vert=vert,
                      showmeans=True, showextrema=False, showmedians=False)
    for i, b in enumerate(v['bodies']):
        if i == 0 and color is None:
            color = b.get_facecolor().flatten()
        if vert:
            # get the center
            m = np.mean(b.get_paths()[0].vertices[:, 0])
            if left:
                # modify the paths to not go further right than the center
                b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], -np.inf, m)
            else: # right
                b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], m, np.inf)
        else:
            # get the center
            m = np.mean(b.get_paths()[0].vertices[:, 1])
            if left:
                b.get_paths()[0].vertices[:, 1] = np.clip(b.get_paths()[0].vertices[:, 1], -np.inf, m)
            else: # right
                b.get_paths()[0].vertices[:, 1] = np.clip(b.get_paths()[0].vertices[:, 1], m, np.inf)
        if color is not None:
            # TODO: error handling
            b.set_color(color)
    return color
