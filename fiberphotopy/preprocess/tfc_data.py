import numpy as np
import pandas as pd
from pathlib import Path
from preprocess.fp_data import trial_normalize, save_data


def make_tfc_comp_times(n_trials, baseline, cs_dur, trace_dur, us_dur, iti_dur):

    comp_times = []
    session_time = 0
    comp_times.append(["baseline", baseline, 0, baseline])
    session_time += baseline
    for t in range(n_trials):
        comp_times.append([f"tone-{t+1}", cs_dur, session_time, session_time + cs_dur])
        session_time += cs_dur
        comp_times.append(
            [f"trace-{t+1}", trace_dur, session_time, session_time + trace_dur]
        )
        session_time += trace_dur
        comp_times.append([f"shock-{t+1}", us_dur, session_time, session_time + us_dur])
        session_time += us_dur
        comp_times.append([f"iti-{t+1}", iti_dur, session_time, session_time + iti_dur])
        session_time += iti_dur

    comp_times_df = pd.DataFrame(
        comp_times, columns=["component", "duration", "start", "end"]
    )

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

    doc_dir = Path(__file__).parents[2] / "docs"
    component_label_file = "TFC phase components.xlsx"

    return pd.read_excel(doc_dir / component_label_file, sheet_name=session)


def find_tfc_components(df, session="train"):
    # TODO: Test function
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


@save_data
def trials_df(
    df,
    session="train",
    yvar="465nm_dFF",
    normalize=True,
    trial_start=-20,
    cs_dur=20,
    trace_dur=20,
    us_dur=2,
    iti_dur=120,
):
    """
    1. Creates a dataframe of "Trial data", from (trial_start, trial_end) around each CS onset
    2. Normalizes dFF for each trial to the avg dFF of each trial's pre-CS period.

    ! Session must be a sheet name in 'TFC phase components.xlsx'

    Parameters
    ----------
    df : DataFrame
        Session data to calculate trial-level data.
    session : str, optional
        Name of session used to label DataFrame, by default 'train'
    yvar : str, optional
        Name of data to trial-normalize, by default '465nm_dFF'
    normalize : bool, optional
        Normalize yvar to baseline of each trial, by default True
    trial_start : int, optional
        Start of trial, by default -20
    cs_dur : int, optional
        CS duration used to calculate trial time, by default 20
    us_dur : int, optional
        US duration, by default 2
    trace_dur : int, optional
        Trace interval duration, by default 20
    iti_dur : int, optional
        Length of inter-trial-interval; used to calculate trial time, by default 120


    Returns
    -------
    DataFrame
        Trial-level data with `yvar` trial-normalized.
    """
    df = label_tfc_phases(df, session=session)

    comp_labs = load_tfc_comp_times(session=session)
    tone_idx = [
        tone
        for tone in range(len(comp_labs["component"]))
        if "tone" in comp_labs["component"][tone]
    ]
    iti_idx = [
        iti
        for iti in range(len(comp_labs["component"]))
        if "iti" in comp_labs["component"][iti]
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
    # normalize data
    if normalize:
        df_list = []
        for animal in df["Animal"].unique():
            df_animal = df.query("Animal == @animal")
            df_list.append(trial_normalize(df_animal, yvar=yvar))
        df = pd.concat(df_list)

    return df


@save_data
def trials_df_new(
    df,
    session="train",
    yvar="465nm_dFF",
    normalize=True,
    trial_baseline=-20,
    cs_dur=20,
    trace_dur=20,
    us_dur=2,
    iti_dur=120,
):
    # TODO: Rework to use only tone_idx (use trace + us_dur + iti)
    # TODO: Test func

    df = label_tfc_phases(df, session=session).copy()

    comp_labs = load_tfc_comp_times(session=session)
    tone_idx = [
        tone
        for tone in range(len(comp_labs["component"]))
        if "tone" in comp_labs["component"][tone]
    ]
    iti_idx = [
        iti
        for iti in range(len(comp_labs["component"]))
        if "iti" in comp_labs["component"][iti]
    ]
    # determine number of tone trials from label
    n_trials = len(tone_idx)
    n_subjects = df.Animal.nunique()
    trial_num = int(1)
    # subset trial data (-20 prior to CS --> 100s after trace/shock)
    for tone, iti in zip(tone_idx, iti_idx):
        start = comp_labs.loc[tone, "start"] + trial_baseline
        end = comp_labs.loc[iti, "start"] + iti_dur + trial_baseline
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
        trial_baseline,
        trial_baseline + cs_dur + trace_dur + us_dur + iti_dur,
        n_trial_pts,
    )
    df["time_trial"] = np.tile(np.tile(time_trial, n_trials), n_subjects)
    # normalize data
    if normalize:
        return trial_normalize(df, yvar=yvar)
    else:
        return df
