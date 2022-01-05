""" Code for PSTH-related statistical analysis."""
import pandas as pd
from scipy import stats

# functions for:
# pre-post t-test (pinguin? scipy?)
# trial epoch RM ANOVA (pinguin)


def calc_pre_post(df, event, t_pre, t_post, measure="mean"):
    """
    Compute the average over a defined pre and post period.
    
    Parameters
    ----------    
    df : DataFrame
        Pandas DataFrame with data to calculate over.
    t_pre: tuple
        Time points for pre-event period (start, end)
    t_post : tuple
        Time points for pre-event period (start, end)
    measure : str, optional
        Specify metric used to calculate pre-post, by default 'mean'.
    
    Returns
    -------
    DataFrame
        Averaged data across the give t_pre and t_post
    """

    df = df.copy()
    df_pre = df[df["time_trial"].between(t_pre[0], t_pre[1])].reset_index(drop=True)
    df_post = df[df["time_trial"].between(t_post[0], t_post[1])].reset_index(drop=True)
    # add `epoch` column
    df_pre["epoch"] = f"pre-{event}"
    df_post["epoch"] = f"post-{event}"
    # recombine values and groupby new epoch var
    df_prepost = pd.concat([df_pre, df_post])

    if measure == "mean":
        return df_prepost.groupby(["Animal", "epoch"]).mean().reset_index()
    elif measure == "max":
        df_prepost = (
            df_prepost.groupby(["Animal", "time_trial", "epoch"]).mean().reset_index()
        )
        return df_prepost.groupby(["Animal", "epoch"]).max().reset_index()

    """
    Compute a paired t-test for pre and post event.

    Parameters
    ----------
    df_prepost : DataFrame
        Output from calc_pre_post
    yvar : str
        Name of independent variable to compare, by default '465nm_dFF_znorm'.
    values : bool, optional
        Return the tstat and pval for the t-test, by default False.

    Returns
    -------
    (tstat, pval)
        Returns the t-statistic and the p-value from the paired t-test.
    """


def pre_post_stats(df_prepost, yvar="465nm_dFF_znorm", return_values=False):
    """
    Compute a paired t-test for pre and post event.

    Args:
        df_prepost (DataFrame): Output of `calc_pre_post`
        yvar (str, optional): Name of dependent variable. Defaults to "465nm_dFF_znorm".
        return_values (bool, optional): [description]. Defaults to False.

    Returns:
        [type]: [description]
    """
    pre = df_prepost.loc[df_prepost["epoch"].str.contains("pre"), yvar]
    post = df_prepost.loc[df_prepost["epoch"].str.contains("post"), yvar]
    tstat, pval = stats.ttest_rel(pre, post)

    print(f" t-statistic: {tstat} \n p-value: {pval}")

    if return_values:
        return (tstat, pval)
