"""Code for loading and cleaning data."""
import datetime
from functools import wraps
from pathlib import Path
from typing import Callable

import numpy as np
from numpy.typing import ArrayLike
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from scipy.optimize import curve_fit


def save_data(func: Callable) -> None:
    """
    Save data decorator function.

    Args:
        func (callable): The function for creating data.

    Returns:
        *args, **kwargs: Arguments & keyword arguments.
    """

    @wraps(func)
    def wrapper(*args: str, **kwargs: str) -> Callable:
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
    filename: str,
    sig_name: str = None,
    ref_name: str = None,
    input_ch: int = 1,
    sig_led: int = 2,
    ref_led: int = 1,
    animal_id: str = None,
) -> pd.DataFrame:
    """
    Load photometry data from Doric Neuroscience Studio.

    Parameters
    ----------
    filename : str
        input filename
    sig_name : str, optional
        Signal channel name, by default None
    ref_name : str, optional
        Reference channel name, by default None
    input_ch : int, optional
        Analog input channel on acquisiton system, by default 1
    sig_led : int, optional
        LED channel for signal channel, by default 2
    ref_led : int, optional
        LED channel for reference channel, by default 1
    animal_id : str, optional
        Animal, by default None

    Returns
    -------
    pd.DataFrame
    """
    df_raw = pd.read_csv(f"{filename}")
    df = df_raw.copy()
    # rename Doric data cols
    df.columns = [col.replace(" ", "") for col in df.columns]
    new_col_names = {
        "Time(s)": "time",
        f"AIn-{input_ch}-Dem(AOut-{ref_led})": ref_name if ref_name else "Y_ref",
        f"AIn-{input_ch}-Dem(AOut-{sig_led})": sig_name if sig_name else "Y_sig",
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
    # df.loc[:, ttl_cols] = df.loc[:, ttl_cols].astype(int)
    df[df.columns[ttl_cols]] = df[df.columns[ttl_cols]].astype(int)
    # drop any TTL channels with all 1s or 0s
    for col in df.loc[:, ttl_cols].columns:
        if len(pd.unique(df.loc[:, col])) == 1:
            df = df.drop(col, axis=1)

    return df


def trim_ttl_data(df: pd.DataFrame, TTL_session_ch: int = 1, TTL_on: int = 0) -> pd.DataFrame:
    """
    Find first and last TTL input (to indicate start and end of behavioral session).
    - In the Doric recording TTL value is 1.
    - When Med-Assocaites SG-231 is ON, TTL value set to 0

    Parameters
    ----------
    df : pd.DataFrame
        Data containing TTL pulses for session start/stop.
    TTL_session_ch : int
        IO channel containing session start/stop TTL pulses, by default 1.
    TTL_on : int
        Value whne TTL pulse is ON, by default 0

    Returns
    -------
        out: pd.DataFrame
            Trimmed DataFrame.
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


def resample_data(df: pd.DataFrame, freq: int) -> pd.DataFrame:
    """
        Resample DataFrame to the provided frequency.


    Parameters
    ----------
    df : pd.DataFrame
        Data to resample.
    freq : int
        Resampling frequency.

    Returns
    -------
    pd.DataFrame
        Resampled data.
    """
    period = 1 / freq

    df = df.copy()
    subject_id = df["Animal"].iloc[0]
    # convert index to timedelta and resample
    df.index = df["time"]
    df.index = pd.to_timedelta(df.index, unit="s")
    # TODO: check behavior of resample and set numeric_only explicitly.
    df = df.resample(f"{period}S").mean(numeric_only=True)
    df["time"] = df.index.total_seconds()
    df = df.reset_index(drop=True)
    # resample also moves 'Animal' to end of DataFrame, put it back at front
    cols = df.columns.tolist()
    cols.insert(0, "Animal")
    df = df.reindex(columns=cols)
    df["Animal"] = subject_id
    # for some reason this function converts TTL cols to float64
    ttl_cols = df.columns.str.contains("ttl")
    df[df.columns[ttl_cols]] = df[df.columns[ttl_cols]].astype(int)

    return df


@save_data
def load_session_data(
    filedir: str | Path,
    sig_name: str = "465nm",
    ref_name: str = "405nm",
    input_ch: int = 1,
    ref_led: int = 1,
    sig_led: int = 2,
    subject_dict: dict[str, str] = None,
    TTL_trim: bool = True,
    TTL_session_ch: int = 1,
    TTL_on: int = 0,
    downsample: bool = True,
    freq: int = 10,
) -> pd.DataFrame:
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
        subject_dict (dict[str, str]): Dict mapping filename to subject names. Defaults to None.
        TTL_trim (bool, optional): align session data with TTL pulse. Defaults to True.
        TTL_session_ch (int, optional): TTL input channel for session start and end. Defaults to 1.
        TTL_on (int, optional): Value of TTL pulse when ON. Defaults to 0.
        downsample (bool, optional): Downsample the data. Defaults to True.
        freq (int, optional): Frequency to downsample data to.. Defaults to 10.

    Returns:
        DataFrame: Combined data for every file in the input directory.
    """
    data_file_list = [str(data_file) for data_file in list(Path(filedir).glob("*.csv"))]
    df_list: list[pd.DataFrame] = []
    for data_file in data_file_list:
        df_temp = load_doric_data(data_file, sig_name, ref_name, input_ch, sig_led, ref_led)
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


def smooth_trial_data(df: pd.DataFrame, yvar: str, smooth_factor: float = 0.025) -> pd.DataFrame:
    """
    Apply LOESS filter to smooth data (for visualization only).

    Args:
        df (DataFrame): Trial-level data
        yvar (str): Name of y-variable to smooth.
        smooth_factor (float, optional): Amount of smoothing to apply. Defaults to 0.025.
    """

    def _smooth_subject_trials(
        df: pd.DataFrame, yvar: str = "dFF_baseline_norm", _smooth_factor: float = 0.025
    ) -> pd.DataFrame:
        lowess = sm.nonparametric.lowess
        dff_smooth = []
        for _trial in df["Trial"].unique():
            df_trial = df.query("Trial == @_trial")
            x = df_trial["time_trial"].values
            y = df_trial[yvar].values
            mod_smooth = lowess(y, x, frac=_smooth_factor, return_sorted=False)
            dff_smooth.extend(mod_smooth)

        return df.assign(dFF_smooth=dff_smooth)

    smooth_data_list = []
    for _subject in df["Animal"].unique():
        df_subj = df.query("Animal == @_subject")
        smooth_data_list.append(
            _smooth_subject_trials(df_subj, yvar=yvar, _smooth_factor=smooth_factor)
        )

    return pd.concat(smooth_data_list)


def trial_normalize(df: pd.DataFrame, yvar: str) -> pd.DataFrame:
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

    znorm_vals = []
    bnorm_vals = []
    for _trial in df["Trial"].unique():
        df_trial = df.query("Trial == @_trial")
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


def get_ols_preds(Y: ArrayLike, X: ArrayLike) -> ArrayLike:
    """Get simple linear reression predictions."""
    mod = sm.OLS(Y, X).fit()
    return mod.predict(X)


def fit_linear(
    df: pd.DataFrame, Y_sig: str, Y_ref: str, by_trial: bool = False, debleached: bool = False
) -> pd.DataFrame:
    """
    Apply linear regression to model fluorescence data.

    Args:
        df (DataFrame): Data containing reference and signal data to model.
        Y_sig (str, optional): Name of signal channel. Defaults to "465nm".
        Y_ref (str, optional): Name of reference channel. Defaults to "405nm".
        by_trial (bool, optional): Apply fit separately on each trial. Defaults to False.
        debleached (bool, optional): Apply debleaching prior to linear fit. Defaults to False.

    Returns:
        pd.DataFrame: Output DataFrame with new column "dFF" or "dFF_debleached".
    """

    df = df.copy()

    if by_trial:
        assert "Trial" in df.columns, "'Trial' column missing from DataFrame"
        new_cols = [x for x in df.columns.to_list() if "nm_" not in x and "dFF" not in x]
        df = df[new_cols].copy()
        Ypred = []
        for trial in df["Trial"].unique():
            X = df.loc[df["Trial"] == trial, Y_ref].values
            Y = df.loc[df["Trial"] == trial, Y_sig].values
            Ypred.extend(get_ols_preds(Y, X))
    else:
        Ypred = get_ols_preds(Y=df[Y_sig], X=df[Y_ref])

    if debleached:
        dFF = df[Y_sig] - Ypred
    else:
        dFF = (df[Y_sig] - Ypred) / Ypred * 100

    return df.assign(
        **{
            f"{Y_sig}_pred": Ypred,
            f"{Y_sig}_dFF": dFF,
            f"{Y_sig}_dFF_zscore": stats.zscore(dFF, ddof=1),
        }
    )


def fit_biexponential(df: pd.DataFrame, t: str, y: str) -> ArrayLike:
    """
    Fit biexponential model.

    Args:
        df (DataFrame): Data to model.
        t (str): Name of time variable.
        y (str): Name of signal to fit.

    Returns:
        ArrayLike: Biexponential model predictions.
    """

    def _biexponential(x: ArrayLike, a: float, b: float, c: float, d: float) -> ArrayLike:
        return a * np.exp(b * x) + c * np.exp(d * x)

    popt, *_ = curve_fit(_biexponential, df[t], df[y], p0=(0.5, 0, 0.5, 0), maxfev=10000)
    biexp = _biexponential(df[t], *popt)

    return biexp


def debleach_signals(
    df: pd.DataFrame, Y_ref: str = "405nm", Y_sig: str = "465nm", by_trial: bool = False
) -> pd.DataFrame:
    """
    Debleach photodecay using biexponential model.

    Args:
        df (DataFrame): Data to debleach.
        Y_ref (str, optional): Name of reference channel. Defaults to "405nm".
        Y_sig (str, optional): Name of signal channel. Defaults to "465nm".
        by_trial (bool, optional): Apply trial-by-trial. Defaults to False.

    Returns:
        DataFrame: Debleached data.
    """

    df = df.copy()

    if by_trial:
        assert "Trial" in df.columns, "'Trial' column missing from DataFrame"
        ref_biexp: list[pd.DataFrame] = []
        sig_biexp: list[pd.DataFrame] = []
        for _trial in df["Trial"].unique():
            df_trial = df.query("Trial == @_trial")
            ref_biexp.extend(fit_biexponential(df_trial, t="time_trial", y=Y_ref))
            sig_biexp.extend(fit_biexponential(df_trial, t="time_trial", y=Y_sig))

    else:
        ref_biexp = fit_biexponential(df, t="time", y=Y_ref)
        sig_biexp = fit_biexponential(df, t="time", y=Y_sig)

    return df.assign(ref_debleach=df[Y_ref] - ref_biexp, sig_debleach=df[Y_sig] - sig_biexp)
