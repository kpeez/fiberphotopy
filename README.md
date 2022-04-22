
# fiberphotopy

Code for analyzing fiber photometry data collected on the Doric Fiber
Photometery acquisition system.

Import the package as follows:

``` {.python}
import fiberphotopy as fp
```

## Installation

The easiest way to install fiberphotopy is with `pip`. First, clone the
repository.

``` {.bash}
git clone https://github.com/kpuhger/fiberphotopy.git
```

Next, navigate to the cloned repo and type the following into your
terminal:

``` {.bash}
pip install .
```

**Note:** The installation method currently is not likely to work.
For the time being it is recommended to add a .pth file to your `site-packages` folder to add the repo to your system's path.

1. Use the terminal to navigate to your `site-packages` folder (e.g., `cd opt/miniconda3/lib/python3.10/site-packages`)
2. Add `.pth` file pointing to repo path

    ```{.bash}
    > touch `fiberphotopy.pth` # create pth file
    > open `fiberphotopy.pth` # add fiberphotopy path to file
    ```

## Features

### Loading data

Whole session data should be stored in a directory and can be loaded
like this:

``` {.python}
fp.load_session_data(...)
```

- Args can be used to modify the name of the signal and reference
    wavelengths as well as to specify the input channel on the
    photoreceiver and the output channel for the two excitation LEDs.
- By default, this function calls `trim_ttl_data` which looks for a
    TTL pulse that indicates the start and end of a behavioral session.
    This is optional and be turned off by setting `TTL_trim=False`.
- By default, this function also downsamples the data to 10 Hz. This
    is controlled by the `downsample=True` argument and the associated
    `freq` argument.
- By default, this function uses the standard method of motion
    correction for photometry data. It fits a linear model to the
    reference channel (e.g., 405nm) to predict the fluoresence in the
    signal channel (e.g., 465nm). Next, it calculates a dFF as:
    `100*(Y-Y_pred)/Y_pred`
- By default, the 'Animal' column will be populated with the name of
    the associated data file. This column can be renamed by creating a
    dict of `{'filename': 'subject_id'}` mappings and passed into
    `load_session_data` with the `subject_dict` argument.

### Visualizing session data

The entire session can be visualized by running:

``` {.python}
fp.plot_fp_session(...)
```

This generates a two-panel plot. The top panel plot the raw reference
and signal fluorescene values on the same plot, and the bottom panel
plots the dFF calculated from those reference and signal values.

### Trial-level data

These functions have only been tested on auditory fear conditioning
experiments (trace or delay FC). Please check the function documentation
for more information.

For trace fear condtioning (TFC) experiments, you can get trial-level
data by calling

``` {.python}
fp.tfc_trials_df(...)
```

- This function takes a DataFrame as input (e.g., from
    `load_session_data`) and creates a trial-level DataFrame with a new
    column 'Trial' containing the trial number and 'time_trial'
    containing a standardized time array for extracting identical events
    across trials.
- By default, this function provides two methods of trial-normalized
    data:
    1. `'dFF_znorm'`: z-score values computed across the entire trial
        period.
    2. `'dFF_baseline_norm'`: baseline-normalized values. Computed as
        (x - mean(baseline))/std(baseline)

### Visualizing trial data

There are 3 main functions to visualize trial-level data:

``` {.python}
fp.plot_trial_avg(...)
```

This will plot the trial-average for the specified yvar. Data is
averaged across trials for each subject, and these subject
trial-averages are used to calculate the trial-level error for plotting.

``` {.python}
fp.plot_trial_indiv(...)
```

This will generate a figure with `m x n` subplots. The shape of the
figure is controlled with the `subplot_params` argument to indicate how
many rows and columns to use for the figure.

``` {.python}
fp.plot_trial_heatmap(...)
```

This will generate a heatmap of the data across trials. If the input
DataFrame contains multiple subjects it will calculate the average
values for each time bin before generating the heatmap.
