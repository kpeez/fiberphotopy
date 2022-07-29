"""Import fiberphotopy."""
# flake8: noqa

__version__ == "0.2.0"


from .plotting.fp_viz import (
    fp_traces_panel,
    plot_dff_data,
    plot_fp_session,
    plot_raw_data,
    plot_single_trial,
    plot_trial_avg,
    plot_trial_heatmap,
    plot_trial_subplot,
)
from .plotting.fp_viz_utils import set_color_palette, set_trialavg_aes
from .preprocess.expt_config import (
    create_expt_config,
    load_expt_config,
    update_expt_config,
)
from .preprocess.fp_data import (
    debleach_signals,
    fit_biexponential,
    fit_linear,
    get_ols_preds,
    load_doric_data,
    load_session_data,
    resample_data,
    trial_normalize,
    trim_ttl_data,
)
from .preprocess.tfc_data import get_tfc_trial_data, make_tfc_comp_times, tfc_trials_df
from .stats.psth_stats import calc_pre_post, pre_post_stats
