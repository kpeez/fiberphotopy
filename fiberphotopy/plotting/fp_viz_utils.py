""" Utilities for plotting functions."""
import os
from functools import wraps
import seaborn as sns
import matplotlib.pyplot as plt


# define color palette:
kp_pal = [
    "#2b88f0",  # blue
    "#EF862E",  # orange
    "#00B9B9",  # cyan
    "#9147B1",  # purple
    "#28A649",  # green
    "#F97B7B",  # salmon
    "#490035",  # violet
    "#bdbdbd",
]  # gray


def set_palette(color_pal=None, show=False):
    """Set default color palette."""
    color_pal = kp_pal if color_pal is None else color_pal
    sns.set_palette(color_pal)
    if show:
        sns.palplot(color_pal)
    else:
        return color_pal


def check_ax(ax, figsize=None):
    """Check whether a figure axes object is defined, define if not.
    Parameters
    ----------
    ax : matplotlib.Axes or None
        Axes object to check if is defined.
    Returns
    -------
    ax : matplotlib.Axes
        Figure axes object to use.
    """

    if not ax:
        _, ax = plt.subplots(figsize=figsize)

    return ax


def savefig(func):
    """Decorator function to save out figures."""

    @wraps(func)
    def decorated(*args, **kwargs):

        save_fig = kwargs.pop("save_fig", False)
        fig_name = kwargs.pop("fig_name", None)
        fig_path = kwargs.pop("fig_path", None)

        func(*args, **kwargs)

        if save_fig:
            if fig_path:
                full_path = os.path.join(fig_path, fig_name)
            else:
                full_path = os.path.expanduser(f"~/Desktop/{fig_name}.png")

            plt.savefig(full_path)

    return decorated


def set_trialavg_aes(ax, title=None, cs_dur=20, us_del=40, us_dur=2):
    """
    Set aesthetics for trialavg plot.

    Parameters
    ----------
    ax : matplotib.axes
        Axes object to apply formatting to
    cs_dur : int, optional
        CS duration (specified in trialavg call), by default 20
    us_del : int, optional
        US delivery time (specified in trialavg call), by default 40
    us_dur : int, optional
        US duration (specified in trialavg call), by default 2

    Returns
    -------
    [type]
        [description]
    """
    # adjust x-axis margin to shift plot adjacent to y-axis
    ax.margins(x=0)
    # add dashed line at y=0, dashed lines for shock
    ax.axhline(y=0, linestyle="-", color="black", linewidth=0.6)
    # add rectangle to highlight CS period
    ax.axvspan(0, cs_dur, facecolor="grey", alpha=0.2)
    # add dashed black rectangle around shock interval
    if us_dur > 0:
        ax.axvspan(
            us_del, us_del + us_dur, facecolor="none", edgecolor="black", ls="--"
        )
    ylab = r"Normalized $\Delta F/F %$"
    xlab = "Time from cue onset (s)"
    tick_size = 22
    label_size = 28
    ax.tick_params(labelsize=tick_size, width=1, length=8)
    ax.set_ylabel(ylab, size=label_size)
    ax.set_xlabel(xlab, size=label_size)

    if title:
        ax.set_title(title)

    return ax
