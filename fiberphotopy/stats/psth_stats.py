"""Code for PSTH-related statistical analysis."""
import pandas as pd
import pingouin as pg

# functions for:
# pre-post t-test (pinguin? scipy?)
# trial epoch RM ANOVA (pinguin)


def calc_pre_post(df, event, t_pre, t_post, measure="mean"):
    """
    Compute the average over a defined pre and post period.

    Args:
        df (DataFrame):
            Pandas DataFrame to calculate pre-post event data.
        t_pre (tuple): Time points for pre-event period (start, end)
        t_post (tuple): Time points for post-event period (start, end)
        measure (str, optional):
            Specify metric used to calculate pre-post. Defaults to 'mean'.

    Returns:
        DataFrame: Averaged data across the give t_pre and t_post
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
        df_prepost = df_prepost.groupby(["Animal", "time_trial", "epoch"]).mean().reset_index()
        return df_prepost.groupby(["Animal", "epoch"]).max().reset_index()


def pre_post_stats(df_prepost, yvar="465nm_dFF_znorm"):
    """
    Compute a paired t-test for pre and post event.

    Args:
        df_prepost (DataFrame): Output of `calc_pre_post`
        yvar (str): Name of dependent variable. Defaults to "465nm_dFF_znorm".

    Returns:
        (tstat, pval) (tuple): the t-statistic and the p-value from the paired t-test.
    """
    pre = df_prepost.loc[df_prepost["epoch"].str.contains("pre"), yvar]
    post = df_prepost.loc[df_prepost["epoch"].str.contains("post"), yvar]

    return pg.ttest(pre, post, paired=True)
