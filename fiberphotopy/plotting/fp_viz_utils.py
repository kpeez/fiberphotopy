""" Utilities for plotting functions."""
import datetime
import inspect
from pathlib import Path
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
    """
    Check whether a figure axes object is defined, define if not.

    Args:
        ax (matplotlib.Axes or None): Axes object to check if is defined.

    Returns:
        matplotlib.Axes: Figure axes object to use.
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
            fig_path = Path(fig_path) if fig_path else Path(Path.home() / "Desktop")

            fig_name = (
                fig_name
                if fig_name
                else datetime.datetime.now().strftime("%Y-%m-%d-%Hh%Mm%Ss-NewFig")
            )

            plt.savefig(f"{fig_path}/{fig_name}.png", facecolor="white", dpi=300)

    return decorated


AXIS_STYLE_ARGS = ["title", "xlabel", "ylabel", "xlim", "ylim"]
# Custom style arguments are those that are custom-handled by the plot style function
CUSTOM_STYLE_ARGS = [
    "title_fontsize",
    "label_size",
    "labelpad",
    "tick_labelsize",
    "legend_size",
    "legend_loc",
    "markerscale",
]
STYLE_ARGS = AXIS_STYLE_ARGS + CUSTOM_STYLE_ARGS
# Define default values for aesthetic
# These are all custom style arguments
TITLE_FONTSIZE = 48
LABEL_PAD = 8
LABEL_SIZE = 24
TICK_LABELSIZE = 24
LEGEND_SIZE = 24
LEGEND_LOC = "best"
MARKERSCALE = 1


def apply_plot_style(ax, style_args=None, **kwargs):
    """
    Apply custom plot style. Used to set default plot options
    """
    style_args = style_args if style_args else AXIS_STYLE_ARGS
    # Apply any provided axis style arguments
    plot_kwargs = {key: val for key, val in kwargs.items() if key in style_args}
    ax.set(**plot_kwargs)
    # update title size
    if ax.get_title():
        ax.set_title(ax.get_title(), fontdict={"fontsize": TITLE_FONTSIZE}, pad=12)
    # Settings for the axis labels and ticks
    label_size = kwargs.pop("label_size", LABEL_SIZE)
    ax.xaxis.label.set_size(label_size * 0.75)
    ax.yaxis.label.set_size(label_size)
    ax.tick_params(
        axis="both",
        which="major",
        pad=kwargs.pop("pad", LABEL_PAD),
        labelsize=kwargs.pop("tick_labelsize", TICK_LABELSIZE),
    )
    # if legend labels get duplicated, pick the original ones
    if ax.get_legend_handles_labels()[0]:
        handles, labels = ax.get_legend_handles_labels()
        nhandles = len(handles)
        first_handle = 0
        ax.legend(
            handles[first_handle:nhandles],
            labels[first_handle:nhandles],
            frameon=False,
            prop={"size": kwargs.pop("legend_size", LEGEND_SIZE)},
            loc=kwargs.pop("legend_loc", LEGEND_LOC),
            markerscale=kwargs.pop("markerscale", MARKERSCALE),
        )

    plt.tight_layout()


def style_plot(func, *args, **kwargs):  # pylint: disable=unused-argument
    """
    Decorator function to make a plot and run apply_plot_style() on it.

    Args:
        func (callable): The plotting function for creating a plot.
        *args, **kwargs: Arguments & keyword arguments.
            These should include any arguments for the plot, and those for applying plot style.
    """

    def get_default_args(func):
        """
        returns a dictionary of arg_name: default_values for the input function
        """
        argspec = inspect.getfullargspec(func)
        return dict(zip(reversed(argspec.args), reversed(argspec.defaults)))

    @wraps(func)
    def decorated(*args, **kwargs):
        # Grab any provided style arguments
        style_args = kwargs.pop("style_args", STYLE_ARGS)
        kwargs_local = get_default_args(func)
        kwargs_local.update(kwargs)
        style_kwargs = {key: kwargs.pop(key) for key in style_args if key in kwargs}
        # Create the plot
        func(*args, **kwargs)
        # Get plot axis, if a specific one was provided, or just grab current and apply style
        cur_ax = kwargs["ax"] if "ax" in kwargs and kwargs["ax"] else plt.gca()

        apply_plot_style(cur_ax, **style_kwargs)

    return decorated


def set_trialavg_aes(ax, title=None, cs_dur=20, us_del=40, us_dur=2):
    """
    Set aesthetics for trialavg plot.

    Args:
        ax (matplotib.Axes): Axes object to apply formatting to
        cs_dur (int, optional): CS duration (specified in trialavg call). Defaults to 20.
    us_del (int, optional): US delivery time (specified in trialavg call). Defaults to 40.
    us_dur : int, optional
        US duration (specified in trialavg call). Defaults to 2.

    Returns:
        matplotlib.Axes
            Figure axes object to use.
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
