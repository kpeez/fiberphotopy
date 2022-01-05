""" Visualize fiber photometry data."""
import numpy as np
import matplotlib.pyplot as plt
from plotting.fp_viz_utils import style_plot
import seaborn as sns
from photometry_analysis import fp_viz_utils, fp_dat
from photometry_analysis.fp_viz_utils import savefig

# define color palette:
kp_pal = [
    "#2b88f0",  # blue
    "#EF862E",  # orange
    "#00B9B9",  # cyan
    "#9147B1",  # purple
    "#28A649",  # green
    "#F97B7B",  # salmon
    "#490035",  # violet
    "#bdbdbd",  # gray
]


def set_palette(color_pal=None, show=False):
    """Set default color palette."""
    color_pal = kp_pal if color_pal is None else color_pal
    sns.set_palette(color_pal)
    if show:
        sns.palplot(color_pal)
    else:
        return color_pal


def plot_style(figure_size=None):
    """Set default plot style."""
    figure_size = figure_size if figure_size else [30, 20]
    size_scalar = (sum(figure_size) / 2) / 25
    # figure and axes info
    plt.rcParams["figure.facecolor"] = "white"
    plt.rc(
        "axes",
        facecolor="white",
        linewidth=2 * size_scalar,
        labelsize=40 * size_scalar,
        titlesize=32 * size_scalar,
        labelpad=5 * size_scalar,
    )

    plt.rc("axes.spines", right=False, top=False)
    # plot-specific info
    plt.rcParams["lines.linewidth"] = 2 * size_scalar
    # tick info
    plt.rcParams["xtick.labelsize"] = 32 * size_scalar
    plt.rcParams["ytick.labelsize"] = 30 * size_scalar
    plt.rcParams["xtick.major.size"] = 10 * size_scalar
    plt.rcParams["ytick.major.size"] = 10 * size_scalar
    plt.rcParams["xtick.major.width"] = 2 * size_scalar
    plt.rcParams["ytick.major.width"] = 2 * size_scalar
    # legend info
    plt.rc("legend", fontsize=32 * size_scalar, frameon=False)


@savefig
@style_plot
def plot_fp_session(
    df,
    yvar="465nm",
    yiso="405nm",
    dffvar="dFF",
    xvar="time",
    session="Training",
    Yiso=True,
    fig_size=(20, 10),
    xlim=None,
    **kwargs,
):
    """
    Plot excitation and isosbestic fluorescence as well as dFF (standard method).

    Parameters
    ----------
    df : DataFrame
        Data to plot. If using trace = 'dFF' DataFrame must contain dFF values.
    yvar : str
        Column containing excitation fluorescence, by default '465nm'
    yiso : str
        Name of column with raw isosbestic values, by default '405nm'
    dffvar : str, optional
        Name of column with dFF values, by default '{yvar}_dFF. Set to None to plot only raw values.
    xvar : str
        Column containing time values, by default 'time'
    session : str, optional
        Name of session, used for figure title and file name if saving, by default 'Training'
    Yiso : bool, optional
        Plot isosbestic signal with excitation, by default True
    fig_size : tuple, optional
        Specify figure size, by default (20, 10)
    xlim : tuple, optional
        Specify x-axis limits, by default None
    save_fig : bool, optional
        Save the figure, by default False. See Notes for more info.

    Notes
    -----
    If using save_fig, can specify a fig_path (default is ~/Desktop).
    """

    # plot aesthetics variables
    tick_size = 18
    label_size = 24
    title_size = 28
    dff_ylab = "z-score" if dffvar == "dFF_zscore" else r"$\Delta F/F$ (%)"

    df = fp_dat.fit_linear(df)

    for idx in df["Animal"].unique():
        df_plot = df.loc[df["Animal"] == idx, :]
        X = df_plot.loc[:, xvar]
        Y = df_plot.loc[:, yvar]
        Yiso = df_plot.loc[:, yiso]
        Ydff = df_plot.loc[:, yvar + "_" + dffvar]

        fig, axs = plt.subplots(2, 1, figsize=fig_size, sharex=True)
        fig.suptitle(f"{session}: {idx}", size=title_size)
        # plot raw fluorescence
        axs[0].plot(X, Y, color=kp_pal[4], label=yvar)
        axs[0].plot(X, Yiso, color=kp_pal[3], label="isosbestic")
        axs[0].set_ylabel("Fluorescence (au)", size=label_size)
        axs[0].legend(fontsize=16, loc="upper right", bbox_to_anchor=(1, 1.1))
        # plot dFF
        axs[1].axhline(y=0, linestyle="-", color="black")
        axs[1].plot(X, Ydff, color=kp_pal[0])
        axs[1].set_xlabel("Time (sec)", size=label_size)
        axs[1].set_ylabel(dff_ylab, size=label_size)

        for ax in axs:
            ax.margins(x=0)
            ax.tick_params(axis="both", labelsize=tick_size, width=2, length=6)

        if xlim:
            plt.xlim(xlim)


@savefig
@style_plot
def plot_traces(
    df,
    yvar="465nm",
    yiso="405nm",
    xvar="time",
    session="Training",
    trace="raw",
    Yiso=True,
    fig_size=(20, 10),
    xlim=None,
    **kwargs,
):
    """
    Plot excitation and isosbestic fluorescence.

    Parameters
    ----------
    df : DataFrame
        Data to plot. If using trace = 'dFF' DataFrame must contain dFF values.
    yvar : str, optional
        Column containing excitation fluorescence, by default '465nm'
    yiso : str, optional
        Name of column with raw isosbestic values, by default '405nm'
    xvar : str, optional
        Column containing time values, by default 'time'
    session : str, optional
        Name of session, used for figure title and file name if saving, by default 'Training'
    trace : str, optional
        Plot raw traces or dFF traces, by default 'raw'
    Yiso : bool, optional
        Plot isosbestic signal with excitation, by default True
    fig_size : tuple, optional
        Specify figure size, by default (20, 10)
    xlim : tuple, optional
        Specify x-axis limits, by default None
    save_fig : bool, optional
        Save the figure, by default False. See Notes for more info.

    Notes
    -----
    If using save_fig, can specify a fig_path (default is ~/Desktop).
    """
    yvar = yvar if trace == "raw" else yvar + f"_{trace}"

    # plot aesthetics variables
    xlab = "Time (sec)"
    ylab = "Fluorescence (au)" if trace == "raw" else r"$\Delta F/F$ (%)"

    tick_size = 18
    label_size = 24
    title_size = 28

    for idx in df["Animal"].unique():
        df_plot = df.loc[df["Animal"] == idx, :]
        fig = plt.figure(figsize=fig_size)
        ax = fig.add_subplot(1, 1, 1)
        # plot aesthetics
        ax.set_ylabel(ylab, size=label_size)
        ax.set_xlabel(xlab, size=label_size)
        ax.margins(x=0)
        if trace == "raw":
            title = f"{session}: {idx}: {yiso} (purple) + {yvar}"
            ax.set_title(title, size=title_size)
            X = df_plot.loc[:, xvar]
            Y = df_plot.loc[:, yvar]
            Yiso = df_plot.loc[:, yiso]
            ax.plot(X, Y, color=kp_pal[4], label=yvar)
            ax.plot(X, Yiso, color=kp_pal[3], label="isosbestic")
            ax.legend(fontsize=16, loc="upper right", bbox_to_anchor=(1, 1.05))

        elif trace in ["dFF", "dFF_zscore"]:
            title = f"{session}: {idx}: {yvar}"
            ax.set_title(title, size=title_size)
            X = df_plot.loc[:, xvar]
            Y = df_plot.loc[:, yvar]
            ax.axhline(y=0, linestyle="-", color="black")
            ax.plot(X, Y, color=kp_pal[4])

        plt.tick_params(axis="both", labelsize=tick_size, width=2, length=6)

        if xlim:
            plt.xlim(xlim)


@savefig
@style_plot
def fp_traces_panel(
    df,
    session="session",
    yvar="465nm",
    yiso="405nm",
    xlim=None,
    y1shift=0.05,
    y1lim=None,
    y2lim=None,
    y3lim=None,
    fig_size=(24, 12),
    **kwargs,
):
    """
    Generate a 3-panel plot:
    1. Mean-centered 465nm/560nm and 405nm fluorescence.
    2. Predicted vs actual 465nm/560nm values.
    3. dFF values from fit_linear()

    Parameters
    ----------
    df : DataFrame
        Data containing raw fluorsence and dFF values.
    session : str
        Name of session. Used to title plot and filename for saving.
    yvar : str, optional
        Name of column with raw fluorescence values, by default '465nm'
    yiso : str, optional
        Name of column with raw isosbestic values, by default '405nm'
    xlim : tuple, optional
        Range to restrict x-axis of plot, by default None
    y1shift : float, optional
        Value to adjust first plot, by default 0.05.
        Use if fluorescence and isosbestic series are still offset after mean-centering.
    y1lim : tuple, optional
        Range to restrict y-axis of plot 1, by default None
    y2lim : tuple, optional
        Range to restrict y-axis of plot 2, by default None
    y3lim : tuple, optional
        Range to restrict y-axis of plot 3, by default None
    fig_size : tuple, optional
        Size of figure, by default (24, 12)
    save_fig : bool, optional
        Save the figure, by default False. See Notes for more info.

    Notes
    -----
    If using save_fig, can specify a fig_path (default is ~/Desktop).
    """
    yiso = "405nm"
    # yvar_col = kp_pal[4]

    for idx in df["Animal"].unique():
        # subset individual animal to plot
        df_plot = df.loc[df["Animal"] == idx, :]
        # generate figures
        fig, ax = plt.subplots(nrows=3, ncols=1, figsize=fig_size)
        # plot-1 405nm and 465nm raw values
        ax[0].plot(
            df_plot["time"],
            df_plot[yiso] - np.mean(df_plot[yiso]),
            # color="purple",
            label="405nm",
        )
        ax[0].plot(
            df_plot["time"],
            df_plot[yvar] - np.mean(df_plot[yvar]) + y1shift,
            # color=yvar_col,
            label=yvar,
        )
        ax[0].legend(loc="upper right", fontsize="small")
        ax[0].set_ylabel("Y-mean(Y)", size=20)
        ax[0].set_title(
            f"mean-centered {yiso[0:5]} (purple) & {yvar} (green) values", size=28
        )
        # plot-1 405nm and 465nm raw values
        ax[1].plot(
            df_plot["time"], df_plot[yvar + "_pred"], color="red", label="predicted"
        )
        ax[1].plot(df_plot["time"], df_plot[yvar], color="black", label="actual")
        ax[1].legend(loc="upper right", fontsize="small")
        ax[1].set_ylabel("Fluorescence (a.u.)", size=20)
        ax[1].set_title(f"{yvar} (black) &  predicted {yvar} (red) values", size=28)
        # dFF values
        ax[2].plot(df_plot["time"], df_plot[yvar + "_dFF"], color="#1b9e77")
        ax[2].set_xlabel("Time (s)")
        ax[2].set_ylabel(r"$\Delta F/F$ (%)", size=20)
        ax[2].set_title(
            r"$\Delta F/F$ values $(\Delta F/F = 100*\frac{Y-\hat{Y}}{\hat{Y}}$)",
            size=28,
        )
        plt.suptitle(f"{session}: Animal {idx}", y=0.99, size=32)
        fig.tight_layout()
        plt.subplots_adjust(top=0.88)
        # adjust xlim and ylims if desired
        for ax in fig.axes:
            ax.set_xlim(xlim)
        yrange_list = [y1lim, y2lim, y3lim]
        for i, ylim in enumerate(yrange_list):
            if ylim:
                fig.axes[i].set_ylim(ylim)


@savefig
@style_plot
def plot_trial_avg(
    df,
    hue=None,
    title=None,
    yvar="465nm_dFF_znorm",
    xvar="time_trial",
    cs_dur=20,
    us_del=40,
    us_dur=2,
    fig_size=(12, 8),
    ax=None,
    **kwargs,
):
    """
        Plot trial-averaged dFF signal.

        Parameters
        ----------
        df : DataFrame
            Trial-level DataFrame from trials_df()
        yvar : str, optional
            Column containing fluorescence values to plot, by default '465nm_dFF_znorm'
        xvar : str, optional
            Column containing trial-level timepoints, by default 'time_trial'
        cs_dur : int, optional
            CS duration. Used to draw rectangle around CS time period, by default 20
        us_del : int, optional
            Time point of US delivery, by default 40
        us_dur : int, optional
            US duration. Used to Draw rectangle around US time period, by default 2
        fig_size : tuple, optional
            Size of figure, by default (12, 8)
        """

    plot_style()
    # initialize the plot and apply trialavg formatting
    ax = fp_viz_utils.check_ax(ax, figsize=fig_size)
    fp_viz_utils.set_trialavg_aes(ax, title, cs_dur, us_del, us_dur)

    if hue:
        hue_means = df.groupby([xvar, hue]).mean().reset_index()
        if hue in ["Animal", "Trial"]:
            hue_stds = df.groupby([xvar, hue]).sem().reset_index()
        else:
            hue_stds = (
                df.groupby([xvar, hue, "Animal"])
                .mean()
                .groupby([xvar, hue])
                .sem()
                .reset_index()
            )
        # plot the data for each hue level
        for hue_level in hue_means[hue].unique():
            x = hue_means.loc[hue_means[hue] == hue_level, xvar]
            y = hue_means.loc[hue_means[hue] == hue_level, yvar]
            yerr = hue_stds.loc[hue_stds[hue] == hue_level, yvar]
            line = ax.plot(x, y, label=f"{hue}: {hue_level}", **kwargs)
            ax.fill_between(
                x, y - yerr, y + yerr, facecolor=line[0].get_color(), alpha=0.15
            )
            ax.legend(fontsize=12)
    else:
        animal_means = df.groupby([xvar]).mean().reset_index()
        animal_stds = (
            df.groupby([xvar, "Animal"]).mean().groupby(xvar).sem().reset_index()
        )
        # grab variables for plotting
        x = animal_means.loc[:, xvar]
        y = animal_means.loc[:, yvar]
        yerror = animal_stds.loc[:, yvar]
        # plot the data
        line = ax.plot(x, y, **kwargs)
        ax.fill_between(
            x, y - yerror, y + yerror, facecolor=line[0].get_color(), alpha=0.15
        )


@savefig
@style_plot
def plot_trial_indiv(
    df, subplot_params=(3, 4), fig_size=(32, 24), suptitle=None, **kwargs
):
    """
    Generate trial-by-trial plot averaged across subjects.
    Users can control the shape of the suplots by passing a tuple into subplot_params.

    Parameters
    ----------
    df : DataFrame
        Trial-level DataFrame from tfc_dat.trials_df()
    subplot_params : tuple, optional
        Shape of subplot (nrows, ncols), by default (3, 4)
    fig_size : tuple, optional
        Size of the figure, by default (38, 24)
    suptitle : [type], optional
        Provide a title for the plot, by default None
    """
    fig, axs = plt.subplots(
        subplot_params[0], subplot_params[1], figsize=fig_size, sharey=False
    )
    xticks = np.arange(int(min(df["time_trial"])), int(max(df["time_trial"])), step=20)
    plt.setp(axs, xticks=xticks, xticklabels=xticks)
    for i, ax in enumerate(axs.reshape(-1)):
        if i + 1 <= max(df["Trial"]):
            single_trial = df.loc[df["Trial"] == i + 1, :]
            plot_trial_avg(single_trial, ax=ax, title=f"Trial {i+1}", **kwargs)
        else:
            fig.delaxes(ax)
    if suptitle:
        plt.suptitle(suptitle, fontsize=40, y=0.995)

    plt.tight_layout(pad=1.5, h_pad=2.5, rect=[0, 0.03, 1, 0.95])


@savefig
@style_plot
def plot_trial_heatmap(
    df, yvar="465nm_dFF_znorm", fig_size=(32, 6), label_size=16, **kwargs
):
    """
    Plot heatmap of dFF across trials.

    Parameters
    ----------
    df : DataFrame
        Trial-level DataFrame from trials_df()
    yvar : str, optional
        Column containing fluorescence values to , by default '465nm_dFF_znorm'
    fig_size : tuple, optional
        Size of figure, by default (32, 6)
    label_size : int, optional
        Size of x-axis tick labels, by default 16
    """
    # pivot df for heatmap format
    df_group_agg = df.pivot_table(
        index="Trial", columns="time_trial", values=yvar, aggfunc="mean"
    )
    plt.figure(1, figsize=fig_size)
    ax = sns.heatmap(
        df_group_agg,
        cbar_kws={"shrink": 0.75, "ticks": None},
        yticklabels=df_group_agg.index,
        **kwargs,
    )
    xlab = "Time from CS onset (sec)"
    ylab = "Trial"
    plt.ylabel(ylab, size=label_size)
    plt.xlabel(xlab, size=label_size)
    # set tick length and remove ticks on y-axis
    ax.tick_params(axis="x", labelsize=label_size, width=2, length=6)
    ax.tick_params(axis="y", which="major", labelsize=label_size, length=0, pad=5)
    # set tick label param size
    ax.tick_params(axis="both", which="major", labelsize=label_size, rotation="auto")
    cbar = ax.collections[0].colorbar
    # here set the labelsize by label_size
    cbar.ax.tick_params(labelsize=label_size, length=0)
    # rescale x-axis into 10-sec labels
    xmin = min(df["time_trial"])
    xmax = max(df["time_trial"])
    xloc = np.arange(0, len(df_group_agg.columns), 50)
    xlabs = np.arange(int(xmin), int(xmax), 5)
    plt.xticks(xloc, xlabs)  # , rotation=45)


# @savefig
# def tfc_trial_avg_old(
#     df,
#     yvar="465nm_dFF_znorm",
#     xvar="time_trial",
#     cs_dur=20,
#     us_del=40,
#     us_dur=2,
#     fig_size=(12, 8),
#     **kwargs,
# ):
#     """
#     Plot trial-averaged dFF signal.

#     Parameters
#     ----------
#     df : DataFrame
#         Trial-level DataFrame from trials_df()
#     yvar : str, optional
#         Column containing fluorescence values to plot, by default '465nm_dFF_znorm'
#     xvar : str, optional
#         Column containing trial-level timepoints, by default 'time_trial'
#     cs_dur : int, optional
#         CS duration. Used to draw rectangle around CS time period, by default 20
#     us_del : int, optional
#         Time point of US delivery, by default 40
#     us_dur : int, optional
#         US duration. Used to Draw rectangle around US time period, by default 2
#     fig_size : tuple, optional
#         Size of figure, by default (12, 8)
#     """

#     # collapse data across all trials
#     mean_vals = (
#         df.loc[:, ["Animal", "Trial", xvar, yvar]].groupby([xvar]).mean().reset_index()
#     )
#     # collapse across trials within each animal
#     avg_subj_trial = (
#         df.loc[:, ["Animal", "Trial", xvar, yvar]]
#         .groupby(["Animal", xvar])
#         .mean()
#         .reset_index()
#     )
#     error_vals = avg_subj_trial.groupby([xvar]).sem().reset_index()
#     # grab variables for plotting
#     X = mean_vals.loc[:, xvar]
#     Y = mean_vals.loc[:, yvar]
#     Yerror = error_vals.loc[:, yvar]
#     # generate figure and add subplot
#     fig = plt.figure(figsize=fig_size)
#     ax = fig.add_subplot(1, 1, 1)
#     # add dashed line at y=0, dashed lines for shock
#     ax.axhline(y=0, linestyle="-", color="black", linewidth=0.6)
#     # add rectangle to highlight CS period
#     ax.axvspan(0, cs_dur, facecolor="grey", alpha=0.2)
#     # add dashed black rectangle around shock interval
#     ax.axvspan(us_del, us_del + us_dur, facecolor="none", edgecolor="black", ls="--")
#     # plot the data
#     plt.plot(X, Y, linewidth=1.5, **kwargs)
#     plt.fill_between(X, Y - Yerror, Y + Yerror, alpha=0.15, **kwargs)
#     # adjust x-axis margin to shift plot adjacent to y-axis
#     ax.margins(x=0)
#     # change label size
#     ylab = r"Normalized $\Delta F/F %$"
#     xlab = "Time from cue onset (s)"
#     # changed from 20,20 to 22,28 on 8-5-2019
#     tick_size = 22
#     label_size = 28
#     ax.tick_params(labelsize=tick_size, width=1, length=8)
#     ax.set_ylabel(ylab, size=label_size)
#     ax.set_xlabel(xlab, size=label_size)
