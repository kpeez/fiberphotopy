"""Process TFC data."""
from pathlib import Path

import numpy as np
import pandas as pd

from .fp_data import debleach_signals, fit_linear, save_data, trial_normalize


def make_tfc_comp_times(n_trials, baseline, cs_dur, trace_dur, us_dur, iti_dur):
    """
    Create component times instead of loading from file.

    Args:
        n_trials (int): number of trials
        baseline (int): duration of baseline periond (in seconds).
        cs_dur (int): CS duration (in seconds).
        trace_dur (int): Trace interval duration (in seconds).
        us_dur (int): Duration of US (in seconds).
        iti_dur (int): Duration of ITI (in seconds).

    Returns:
        DataFrame: Component times
    """
    comp_times = []
    session_time = 0
    comp_times.append(["baseline", baseline, 0, baseline])
    session_time += baseline
    for t in range(n_trials):
        comp_times.append([f"tone-{t+1}", cs_dur, session_time, session_time + cs_dur])
        session_time += cs_dur
        comp_times.append([f"trace-{t+1}", trace_dur, session_time, session_time + trace_dur])
        session_time += trace_dur
        comp_times.append([f"shock-{t+1}", us_dur, session_time, session_time + us_dur])
        session_time += us_dur
        comp_times.append([f"iti-{t+1}", iti_dur, session_time, session_time + iti_dur])
        session_time += iti_dur

    comp_times_df = pd.DataFrame(comp_times, columns=["component", "duration", "start", "end"])

    # clean up component times df for trial order
    new_comps = ["baseline"]
    for comp in comp_times_df["component"][1:]:
        if int(comp.split("-")[-1]) < 10:
            new_comps.append("-0".join(comp.split("-")))
        else:
            new_comps.append("".join(comp))

    comp_times_df["component"] = new_comps
    # only keep components with non-zero duration
    comp_times_df[comp_times_df["duration"] != 0].reset_index(drop=True)

    return comp_times_df


def load_tfc_comp_times(session="train"):
    """
    Load TFC phase components.xlsx from /docs.

    Args:
        session (str, optional): Name of session to load. Defaults to "train".

    Returns:
        DataFrame: Component data frame.
    """
    doc_dir = Path(__file__).parents[2] / "docs"
    component_label_file = "TFC phase components.xlsx"

    return pd.read_excel(doc_dir / component_label_file, sheet_name=session)


def find_tfc_components(df, session="train"):
    # TODO: Test function
    """
    Find TFC components from TFC phase components.xlsx file.

    Args:
        df (DataFrame): DataFrame to get components for
        session (str, optional): Session to get components from. Defaults to "train".

    Returns:
        DataFrame: Data with component labels added.
    """
    comp_labs = load_tfc_comp_times(session=session)
    session_end = max(comp_labs["end"])
    df_new = df.drop(df[df["time"] >= session_end].index)
    # search for time in sec, index into comp_labels
    # for start and end times
    for i in range(len(comp_labs["component"])):
        df_new.loc[
            df_new["time"].between(comp_labs["start"][i], comp_labs["end"][i]),
            "Component",
        ] = comp_labs["component"][i]

    return df_new


def label_tfc_phases(df, session="train"):
    # TODO: Test function
    """
    Label TFC phases using TFC components. "Phases" are simply aggregated components of same type.

    Args:
        df (DataFrame): DataFrame to label.
        session (str, optional): Session to label phases. Defaults to "train".

    Returns:
        DataFrame: Data with Phase labels added.
    """
    session_list = [
        "train",
        "tone",
        "ctx",
        "extinction",
        "cs_response",
        "shock_response",
    ]
    session_type = [sesh for sesh in session_list if sesh in session][0]
    df = find_tfc_components(df, session=session_type)
    df.loc[:, "Phase"] = df.loc[:, "Component"]
    # label tone, trace, and iti for all protocols
    df.loc[df["Phase"].str.contains("tone"), "Phase"] = "tone"
    df.loc[df["Phase"].str.contains("trace"), "Phase"] = "trace"
    df.loc[df["Phase"].str.contains("iti"), "Phase"] = "iti"
    # label shock phases for training data
    df.loc[df["Phase"].str.contains("shock"), "Phase"] = "shock"

    return df


def get_tfc_trial_data(
    df,
    session,
    trial_start,
    cs_dur,
    trace_dur,
    us_dur,
    iti_dur,
):
    """
    1. Creates a dataframe of "Trial data", from (trial_start, trial_end) around each CS onset.
    2. Normalizes dFF for each trial to the avg dFF of each trial's pre-CS period.

    ! Note: Session must be a sheet name in 'TFC phase components.xlsx'

    Args:
        df (DataFrame): Session data to be converted to trial-level format.
        session (str): Name of session used to label DataFrame. Defaults to "train".
        yvar (str): Name of dependent variable to trial-normalize. Defaults to "465nm_dFF".
        normalize (bool, optional): Normalize yvar to baseline of each trial. Defaults to True.
        trial_start (int, optional): Time at start of trial. Defaults to -20.
        cs_dur (int, optional): CS duration used to calculate trial time. Defaults to 20.
        trace_dur (int, optional): Duration of trace interval. Defaults to 20.
            Set to 0 for delay conditioning.
        us_dur (int, optional): Duration of unconditional stimulus. Defaults to 2.
        iti_dur (int, optional): Duration of intertrial interval. Defaults to 120.

    Returns:
        DataFrame: Trial-level data with `yvar` trial-normalized.
    """
    df = label_tfc_phases(df, session=session)
    comp_labs = load_tfc_comp_times(session=session)

    tone_idx = [
        tone
        for tone in range(len(comp_labs["component"]))
        if "tone" in comp_labs["component"][tone]
    ]
    iti_idx = [
        iti for iti in range(len(comp_labs["component"])) if "iti" in comp_labs["component"][iti]
    ]
    # determine number of tone trials from label
    n_trials = len(tone_idx)
    n_subjects = df.Animal.nunique()
    trial_num = int(1)
    # subset trial data (-20 prior to CS --> 100s after trace/shock)
    for tone, iti in zip(tone_idx, iti_idx):
        start = comp_labs.loc[tone, "start"] + trial_start
        end = comp_labs.loc[iti, "start"] + iti_dur + trial_start
        df.loc[(start <= df.time) & (df.time < end), "Trial"] = int(trial_num)
        trial_num += 1
    # remove extra time points
    df = df.dropna().reset_index(drop=True)
    # check if last_trial contains extra rows and if so, drop them
    first_trial = df.query("Trial == Trial.unique()[0]")
    last_trial = df.query("Trial == Trial.unique()[-1]")
    extra_row_cnt = last_trial.shape[0] - first_trial.shape[0]
    df = df[:-extra_row_cnt] if extra_row_cnt > 0 else df
    df.loc[:, "Trial"] = df.loc[:, "Trial"].astype(int)
    # create common time_trial
    n_trial_pts = len(df.query("Animal == Animal[0] and Trial == Trial[0]"))
    time_trial = np.linspace(
        trial_start, trial_start + cs_dur + trace_dur + us_dur + iti_dur, n_trial_pts
    )
    df["time_trial"] = np.tile(np.tile(time_trial, n_trials), n_subjects)

    return df


@save_data
def tfc_trials_df(
    session_df,
    session="train",
    yvar="465nm_dFF",
    trial_dff=False,
    trial_debleach=False,
    trial_start=-20,
    cs_dur=20,
    trace_dur=20,
    us_dur=2,
    iti_dur=120,
):
    """
    1. Creates a dataframe of "Trial data", from (trial_start, trial_end) around each cue onset.
    2. Normalizes dFF for each trial to the avg dFF of each trial's baseline period (pre-cue).

    Args:
        df (DataFrame): Session data to be converted to trial-level format.
        session (str): Name of session used to label DataFrame. Defaults to "train".
        yvar (str): Name of dependent variable to trial-normalize. Defaults to "465nm_dFF".
        trial_dff (bool, optional): Fit trial_dFF on each trial. Defaults to False.
        trial_debleach (bool, optional): Fit biexponential on each trial. Defaults to False.
        trial_start (int): _description_. Defaults to -20.
        cs_dur (int): CS duration used to calculate trial time. Defaults to 20.
        trace_dur (int): Duration of trace interval. Defaults to 20. Set to 0 for delay FC.
        us_dur (int): Duration of unconditional stimulus. Defaults to 2.
        iti_dur (int): Duration of intertrial interval. Defaults to 120.

    Returns:
        DataFrame: Trial-level data with `yvar` trial-normalized.

    Notes:
    - To use trial-level debleaching, you need to also set trial_dff = True.
    """
    df = get_tfc_trial_data(
        session_df,
        session=session,
        trial_start=trial_start,
        cs_dur=cs_dur,
        trace_dur=trace_dur,
        us_dur=us_dur,
        iti_dur=iti_dur,
    )

    df_list = []
    for animal in df["Animal"].unique():
        if trial_dff:
            df_animal = df.query("Animal == @animal").copy()
            debleach_bool = False
            _Y_ref = "405nm"
            _Y_sig = "465nm"
            yvar = "465nm_dFF"

            if trial_debleach:
                df_animal = debleach_signals(df_animal, by_trial=True)
                debleach_bool = True
                _Y_ref = "ref_debleach"
                _Y_sig = "sig_debleach"
                yvar = "sig_debleach_dFF"

            df_animal = fit_linear(
                df_animal,
                Y_ref=_Y_ref,
                Y_sig=_Y_sig,
                by_trial=True,
                debleached=debleach_bool,
            )

        else:
            df_animal = df.query("Animal == @animal").copy()

        df_list.append(trial_normalize(df_animal, yvar))

    return pd.concat(df_list)
