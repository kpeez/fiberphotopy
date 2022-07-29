"""Visualize fiber photometry data."""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from ..preprocess.fp_data import smooth_trial_data
from .fp_viz_utils import _make_ax, savefig, set_trialavg_aes, style_plot

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


@style_plot
def plot_raw_data(
    df_plot,
    yvar="465nm",
    yiso="405nm",
    xvar="time",
    session="Training",
    ax=None,
    **kwargs,
):
    """Plot raw data (`yvar` and `yiso`) over time interval specified in `xvar`."""
    ax = _make_ax(ax)
    X = df_plot.loc[:, xvar]
    Y = df_plot.loc[:, yvar]
    Yiso = df_plot.loc[:, yiso]
    # plot raw fluorescence
    ax.plot(X, Y, color=kp_pal[4], label=yvar)
    ax.plot(X, Yiso, color=kp_pal[3], label="isosbestic")
    ax.set_ylabel("Fluorescence (au)")


@style_plot
def plot_dff_data(
    df_plot,
    xvar="time",
    yvar="465nm",
    dffvar="dFF",
    session="Training",
    ax=None,
    **kwargs,
):
    """Plot dFF data over time interval specified in `xvar`."""
    ax = _make_ax(ax)
    # plot dFF
    X = df_plot.loc[:, xvar]
    Ydff = df_plot.loc[:, yvar + "_" + dffvar]
    dff_ylab = "z-score" if dffvar == "dFF_zscore" else r"$\Delta F/F$ (%)"
    ax.axhline(y=0, linestyle="-", color="black")
    ax.plot(X, Ydff, color=kp_pal[0])
    ax.set_xlabel("Time (sec)")
    ax.set_ylabel(dff_ylab)


def plot_fp_session(
    df,
    yvar="465nm",
    yiso="405nm",
    dffvar="dFF",
    xvar="time",
    session="Training",
    Yiso=True,
    fig_size=(20, 10),
    **kwargs,
):
    """
    Generate a 2-panel plot of data fiber photometry recording session.

    Args:
        df (DataFrame): session data to plot. Can contain several
        yvar (str, optional): Column name of . Defaults to "465nm".
        yiso (str, optional): _description_. Defaults to "405nm".
        dffvar (str, optional): _description_. Defaults to "dFF".
        xvar (str, optional): _description_. Defaults to "time".
        session (str, optional): _description_. Defaults to "Training".
        Yiso (bool, optional): _description_. Defaults to True.
        fig_size (tuple, optional): _description_. Defaults to (20, 10).
    """
    # plot session for each subject
    @savefig
    def _session_plot(
        df_plot,
        **kwargs,
    ):

        fig, axs = plt.subplots(2, 1, figsize=fig_size)  # , sharex=True)
        fig.suptitle(f"{session}: {df_plot['Animal'].unique()[0]}", size=28)
        plot_raw_data(df_plot, ax=axs[0], **kwargs)
        axs[0].legend(fontsize=16, loc="upper right", bbox_to_anchor=(1, 1.1))
        plot_dff_data(df_plot, ax=axs[1], **kwargs)
        for ax in axs:
            ax.margins(x=0)
            ax.tick_params(axis="both", labelsize=18, width=2, length=6)

    for subject in df["Animal"].unique():
        subject_data = df.loc[df["Animal"] == subject, :]
        _session_plot(subject_data, fig_name=f"{session} - {subject} session plot", **kwargs)


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
):
    """
    Plot raw (centered), predicted, and actual dFF traces.

    Generates a 3-panel plot with the following panels:
    1. Mean-centered 465nm/560nm and 405nm fluorescence.
    2. Predicted vs actual 465nm/560nm values.
    3. dFF values from fit_linear()
    """
    yiso = "405nm"

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
        ax[0].set_title(f"mean-centered {yiso[0:5]} (purple) & {yvar} (green) values", size=28)
        # plot-1 405nm and 465nm raw values
        ax[1].plot(df_plot["time"], df_plot[yvar + "_pred"], color="red", label="predicted")
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
    yvar="dFF_baseline_norm",
    xvar="time_trial",
    hue=None,
    smooth=True,
    cs_dur=20,
    us_del=40,
    us_dur=2,
    title=None,
    fig_size=(12, 8),
    ax=None,
    **kwargs,
):
    """
    Plot trial-averaged dFF signal.

    Args:
        df (DataFrame): Trial-level data to visualize.
        yvar (str): Dependent variable to plot. Defaults to "dFF_baseline_norm".
        xvar (str): Name of variable with trial time bins. Defaults to "time_trial".
        hue (str): Specify distinct groups for separate trial-average plots. Defaults to None.
        smooth (bool, optional): Use loess to smooth the data. Defaults to True.
        cs_dur (int): Duration of CS in seconds. Defaults to 20.
        us_del (int): Time of US delivery relative to CS onset. Defaults to 40.
        us_dur (int): Duration of the US in seconds. Defaults to 2.
        title (str, optional): Name of figure. Defaults to None.
        fig_size (tuple, optional): Figure size. Defaults to (12, 8).
        ax (matplotlib.axes.Axes, optional): Specify axes object to add plot to. Defaults to None.
                                             If `None` is provided, `_make_ax` is called.
    """

    plot_style()
    # initialize the plot and apply trialavg formatting
    ax = _make_ax(ax, figsize=fig_size)
    set_trialavg_aes(ax, title, cs_dur, us_del, us_dur)
    kwargs["lw"] = 4

    if smooth:
        # Smooth data with LOESS for plotting
        df = smooth_trial_data(df, yvar=yvar)
        yvar = "dFF_smooth"

    if hue:
        hue_means = df.groupby([xvar, hue]).mean().reset_index()
        if hue in ["Animal", "Trial"]:
            hue_stds = df.groupby([xvar, hue]).sem().reset_index()
        else:
            hue_stds = (
                df.groupby([xvar, hue, "Animal"]).mean().groupby([xvar, hue]).sem().reset_index()
            )
        # plot the data for each hue level
        for hue_level in hue_means[hue].unique():
            x = hue_means.loc[hue_means[hue] == hue_level, xvar]
            y = hue_means.loc[hue_means[hue] == hue_level, yvar]
            yerr = hue_stds.loc[hue_stds[hue] == hue_level, yvar]
            line = ax.plot(x, y, label=f"{hue}: {hue_level}", **kwargs)
            ax.fill_between(x, y - yerr, y + yerr, facecolor=line[0].get_color(), alpha=0.15)
            ax.legend(fontsize=12)
    else:
        animal_means = df.groupby([xvar]).mean().reset_index()
        animal_stds = df.groupby([xvar, "Animal"]).mean().groupby(xvar).sem().reset_index()
        # grab variables for plotting
        x = animal_means.loc[:, xvar]
        y = animal_means.loc[:, yvar]
        yerror = animal_stds.loc[:, yvar]
        # plot the data
        line = ax.plot(x, y, **kwargs)
        ax.fill_between(x, y - yerror, y + yerror, facecolor=line[0].get_color(), alpha=0.15)


@savefig
@style_plot
def plot_trial_subplot(
    df,
    yvar="dFF_znorm",
    xvar="time_trial",
    subplot_dims=(3, 4),
    fig_size=(32, 24),
    suptitle=None,
    **kwargs,
):
    """
    Generate trial-by-trial plot averaged across subjects.
    Users can control the shape of the suplots by passing a tuple into subplot_dims.

    Args:
        df (DataFrame): Trial-level data to plot.
        yvar (str): Dependent variable to plot. Defaults to "dFF_znorm".
        xvar (str): Name of variable with trial time bins. Defaults to "time_trial".
        subplot_dims (tuple, optional): specified row x col. Defaults to (3, 4).
        fig_size (tuple, optional): Figure size. Defaults to (32, 24).
        suptitle (str, optional): Figure title. Defaults to None.
    """
    fig, axs = plt.subplots(subplot_dims[0], subplot_dims[1], figsize=fig_size, sharey=False)
    xticks = np.arange(int(min(df["time_trial"])), int(max(df["time_trial"])), step=20)
    plt.setp(axs, xticks=xticks, xticklabels=xticks)
    for i, ax in enumerate(axs.reshape(-1)):
        if i + 1 <= df["Trial"].max():
            single_trial = df.loc[df["Trial"] == i + 1, :]
            plot_trial_avg(single_trial, yvar, xvar, ax=ax, title=f"Trial {i+1}", **kwargs)
        else:
            fig.delaxes(ax)
    if suptitle:
        plt.suptitle(suptitle, fontsize=40, y=0.995)

    plt.tight_layout(pad=1.5, h_pad=2.5, rect=[0, 0.03, 1, 0.95])


@savefig
@style_plot
def plot_trial_heatmap(df, yvar="dFF_znorm", fig_size=(32, 6), label_size=16, **kwargs):
    """Plot heatmap of dFF across trials."""
    # pivot df for heatmap format
    df_group_agg = df.pivot_table(index="Trial", columns="time_trial", values=yvar, aggfunc="mean")
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
    plt.xticks(xloc, xlabs)


def plot_single_trial(
    trials_df,
    subject,
    trial,
    signals,
    cs_dur=20,
    us_del=40,
    us_dur=2,
):
    """
    Visualize a single trial.

    Args:
        trials_df (DataFrame): Trial data to plot.
        subject (str): Subject id
        trial (int): Trial to plot
        signals (list): List of signals to plot (overlap on same plot).
        cs_dur (int, optional): CS duration. Defaults to 20.
        us_del (int, optional): Time of US delivery. Defaults to 40.
        us_dur (int, optional): US duration. Defaults to 2.
    """
    trial_data = trials_df.query("Animal == @subject and Trial == @trial")

    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    set_trialavg_aes(ax, None, cs_dur, us_del, us_dur)

    for sig in signals:
        sig_centered = trial_data[sig] - trial_data[sig].mean()
        ax.plot(trial_data["time_trial"], sig_centered, label=sig)

    ax.set_title(f"Subject: {subject} Trial: {trial}", pad=20)
    ax.set_ylabel("Fluorescence (au)")
    ax.legend(fontsize=24)
    fig.tight_layout()
