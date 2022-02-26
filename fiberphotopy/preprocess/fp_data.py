"""Code for loading and cleaning data."""
import datetime
from pathlib import Path
from functools import wraps
import numpy as np
import pandas as pd
import statsmodels.api as sm
import scipy.stats as stats


def save_data(func):
    """
    Decorator function for saving out data

    Args:
        func (callable): The function for creating data.

    Returns:
        *args, **kwargs: Arguments & keyword arguments.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        # add save_path and filename kwargs
        save = kwargs.pop("save", False)
        data_path = kwargs.pop("data_path", None)
        filename = kwargs.pop("filename", None)

        if save:
            # set path
            dpath = data_path if data_path else str(Path.cwd())
            # set filename
            filename = (
                filename
                if filename
                else datetime.datetime.now().strftime("%Y-%m-%d-%Hh%Mm%Ss_data")
            )

            func(*args, **kwargs).to_csv(f"{dpath}/{filename}.csv", index=False)

        return func(*args, **kwargs)

    return wrapper


@save_data
def load_doric_data(
    filename,
    sig_name=None,
    ref_name=None,
    input_ch=1,
    sig_led=2,
    ref_led=1,
    animal_id=None,
):
    # TODO: add docstring
    df_raw = pd.read_csv(f"{filename}")
    df = df_raw.copy()
    # rename Doric data cols
    df.columns = [col.replace(" ", "") for col in df.columns]
    new_col_names = {
        "Time(s)": "time",
        f"AIn-{input_ch}-Dem(AOut-{ref_led})": ref_name if ref_name else "reference",
        f"AIn-{input_ch}-Dem(AOut-{sig_led})": sig_name if sig_name else "signal",
    }
    for col in df.columns.to_list():
        if "DI/O" in col:
            new_col_names[col] = "ttl_" + col.split("-")[1]
    df.rename(columns=new_col_names, inplace=True)
    df = df[new_col_names.values()]
    df.insert(0, "Animal", animal_id if animal_id else Path(filename).name[:-4])
    # clean up TTL cols
    ttl_cols = df.columns.str.contains("ttl")
    df.loc[:, ttl_cols] = np.round(df.loc[:, ttl_cols])
    df.loc[:, ttl_cols] = df.loc[:, ttl_cols].astype(int)
    # drop any TTL channels with all 1s or 0s
    for col in df.loc[:, ttl_cols].columns:
        if len(pd.unique(df.loc[:, col])) == 1:
            df = df.drop(col, axis=1)

    return df


def trim_ttl_data(df, TTL_session_ch=1, TTL_on=0):
    """
    Find first and last TTL input (to indicate start and end of behavioral session).
    - In the Doric recording TTL value is 1.
    - When Med-Assocaites SG-231 is ON, TTL value set to 0

    Args:
        df (DataFrame): Data containing TTL pulses for session start/stop.
        TTL_session_ch (int): IO channel containing session start/stop TTL pulses. Defaults to 1.
        TTL_on (int): Value whne TTL pulse is ON. Defaults to 0.

    Returns:
        DataFrame: DataFrame trimmed to session start/stop.
    """
    df = df.copy()
    ttl_ch = "ttl_" + str(TTL_session_ch)
    first_row = min(df[df[ttl_ch] == TTL_on].index)
    last_row = max(df[df[ttl_ch] == TTL_on].index)
    df = df[(df.index >= first_row) & (df.index <= last_row)]
    df = df.reset_index(drop=True)
    # reset 'time'
    df["time"] = df["time"] - df["time"][0]
    # trim DataFrame after shiffting since t0 is now 0.0
    df = df[df["time"] < int(max(df["time"]))].reset_index(drop=True)

    return df


def resample_data(df, freq):
    """
    Resample DataFrame to the provided frequency.

    Args:
        df (DataFrame): DataFrame to resample
        freq (int): New frequency of DataFrame

    Returns:
        DataFrame: Resampled DataFrame
    """
    period = 1 / freq

    df = df.copy()
    subject_id = df["Animal"].iloc[0]
    # convert index to timedelta and resample
    df.index = df["time"]
    df.index = pd.to_timedelta(df.index, unit="s")
    df = df.resample(f"{period}S").mean()
    df["time"] = df.index.total_seconds()
    df = df.reset_index(drop=True)
    # resample also moves 'Animal' to end of DataFrame, put it back at front
    cols = df.columns.tolist()
    cols.insert(0, "Animal")
    df = df.reindex(columns=cols)
    df["Animal"] = subject_id
    # for some reason this function converts TTL cols to float64
    ttl_cols = df.columns.str.contains("ttl")
    df.loc[:, ttl_cols] = df.loc[:, ttl_cols].astype(int)

    return df


@save_data
def load_session_data(
    filedir,
    sig_name="465nm",
    ref_name="405nm",
    input_ch=1,
    ref_led=1,
    sig_led=2,
    subject_dict=None,
    TTL_trim=True,
    TTL_session_ch=1,
    TTL_on=0,
    downsample=True,
    freq=10,
):
    """
    1. Load photometry session data from a directory.
    2. (optional) Trim data to session with TTL pulse.
    3. (optional) Downsample the data.


    Args:
        filedir (str or Path): directory of photometry data files
        sig_name (str, optional): Signal excitation wavelength. Defaults to "465nm".
        ref_name (str, optional): Isosbestic excitation wavelength. Defaults to "405nm".
        input_ch (int): Analong input ch on Doric system. Defaults to 1.
        ref_led (int): Analog output ch for reference channel. Defaults to 1.
        sig_led (int): Analog output ch for signal channel. Defaults to 2.
        TTL_trim (bool, optional): Run `trim_ttl_data` to align to a session TTL pulse. Defaults to True.
        TTL_session_ch (int, optional): TTL input channel for session start and end. Defaults to 1.
        TTL_on (int, optional): Value of TTL pulse when ON. Defaults to 0.
        downsample (bool, optional): Downsample the data. Defaults to True.
        freq (int, optional): Frequency to downsample data to.. Defaults to 10.

    Returns:
        DataFrame: Combined data for every file in the input directory.
    """
    data_file_list = [str(data_file) for data_file in list(Path(filedir).glob("*.csv"))]
    df_list = []
    for data_file in data_file_list:
        df_temp = load_doric_data(
            data_file, sig_name, ref_name, input_ch, sig_led, ref_led
        )
        if TTL_trim:
            df_temp = trim_ttl_data(df_temp, TTL_session_ch, TTL_on)
        if downsample:
            df_temp = resample_data(df_temp, freq)

        df_temp = fit_linear(df_temp, Y_sig=sig_name, Y_ref=ref_name)
        df_list.append(df_temp)

        df = pd.concat(df_list)
        if subject_dict:
            for key in subject_dict:
                subj_rows = df["Animal"].str.contains(key)
                df.loc[subj_rows, "Animal"] = subject_dict[key]

    return df


def get_ols_preds(Y, X):
    """ Get simple linear regression predictions. """
    mod = sm.OLS(Y, X).fit()
    return mod.predict(X)


def get_trial_ols_preds(df, Y, X):
    """ Get trial-level simple linear regression predictions. """

    assert "Trial" in df.columns, "'Trial' column missing from DataFrame"

    new_cols = [x for x in df.columns.to_list() if "nm_" not in x and "dFF" not in x]
    df = df[new_cols].copy()

    Ypred_trial = []
    for trial in df["Trial"].unique():
        X = df.loc[df["Trial"] == trial, X]
        Y = df.loc[df["Trial"] == trial, Y]
        Ypred_trial.extend(get_ols_preds(X, Y))

    return Ypred_trial


def fit_linear(df, Y_sig="465nm", Y_ref="405nm", by_trial=False):

    df = df.copy()
    # X = df[Y_ref]
    # Y = df[Y_sig]
    if by_trial:
        assert "Trial" in df.columns, "'Trial' column missing from DataFrame"
        new_cols = [
            x for x in df.columns.to_list() if "nm_" not in x and "dFF" not in x
        ]
        df = df[new_cols].copy()
        Ypred = []
        for trial in df["Trial"].unique():
            X = df.loc[df["Trial"] == trial, Y_ref].values
            Y = df.loc[df["Trial"] == trial, Y_sig].values
            Ypred.extend(get_ols_preds(Y, X))
    else:
        Ypred = get_ols_preds(Y=df[Y_sig], X=df[Y_ref])

    dFF = (df[Y_sig] - Ypred) / Ypred * 100

    return df.assign(
        **{
            f"{Y_sig}_pred": Ypred,
            f"{Y_sig}_dFF": dFF,
            f"{Y_sig}_dFF_zscore": stats.zscore(dFF, ddof=1),
        }
    )


def trial_normalize(df, yvar):
    """
    Compute a normalized yvar from trial-level data.
    Calculated in two different ways:
        - znorm: whole-trial zscore
        - baseline_norm: standardized to trial baseline.

    Args:
        df (DataFrame): Trial-level data.
        yvar (str): Variable to normalize.

    Returns:
        DataFrame: New column named {yvar}_norm and {yvar}_baseline_norm
    """

    assert "Trial" in df.columns, "'Trial' column missing from DataFrame"
    # df = df.copy()

    znorm_vals = []
    bnorm_vals = []
    for trial in df["Trial"].unique():
        df_trial = df.query("Trial == @trial")
        trial_vals = df_trial[yvar].values
        znorm_vals.append(stats.zscore(trial_vals, ddof=1))
        # trial baseline normalization
        trial_mean = df_trial.loc[df_trial["time_trial"] < 0, yvar].mean()
        trial_std = df_trial.loc[df_trial["time_trial"] < 0, yvar].std()
        bnorm_vals.append((trial_vals - trial_mean) / trial_std)

    return df.assign(
        dFF_znorm=np.asarray(znorm_vals).flatten(),
        dFF_baseline_norm=np.asarray(bnorm_vals).flatten(),
    )
