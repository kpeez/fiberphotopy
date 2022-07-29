"""Code for making expt_config files to describe experimental info."""
import pathlib
from pathlib import Path

import yaml
from ruamel.yaml.main import round_trip_dump as yaml_dump
from ruamel.yaml.main import round_trip_load as yaml_load


def create_expt_config(expt_name=None, config_name=None, project_path=None):
    """
    Create experiment configuration files.

    Args:
        project_name (str): Project name stored in expt_config["project_name"]
        project_path (str, optional): Path to create file in. Defaults to cwd
        config_name (str, optional): Prefix for expt_config file. Defaults to None.

    Returns:
        YAML config file named "expt_config-{config_name}.yml".
    """
    # make config file name
    config_filename = "expt_config-" + config_name if config_name else "expt_config"
    config_file = config_filename + ".yml"
    project_path = project_path if project_path else Path.cwd()
    # test project_path type
    assert isinstance(project_path, pathlib.PurePath) or isinstance(
        project_path, str
    ), "project path must be a str or Path object"
    # test config_filename exists
    for f in list(project_path.glob("*.y*ml")):
        assert f.stem != config_filename, f"Warning! {config_filename} file already exists"
    # create dirs
    for d in ["figures", "data/raw", "data/processed"]:
        Path(str(project_path) + "/" + d).mkdir(parents=True, exist_ok=True)

    # config file info
    new_config = {
        "experiment": expt_name if expt_name else "newproject",
        "dirs": {
            "project": str(project_path),
            "figures": str(project_path) + "/figures",
            "data": str(project_path) + "/data",
        },
        "sessions": {},
        "group_ids": {"group1": list(), "group2": list()},
        "sex_ids": None,
    }
    # lazy convert dict with ruamel
    config_dict = yaml_load(yaml_dump(new_config), preserve_quotes=True)
    # add comments to config
    config_dict.yaml_set_comment_before_after_key(
        key="project_path", before="Set project and fig paths"
    )
    config_dict.yaml_set_comment_before_after_key(key="sessions", before="Session info")
    config_dict.yaml_set_comment_before_after_key(key="group_ids", before="Group info")
    # dump yaml into new file
    with open(f"{project_path / config_file}", "w") as f:
        f.write(yaml_dump(config_dict))

    print(f"{config_file} has been created in \n {project_path}")


def load_expt_config(config_path):
    """
    Load expt_info.yml to obtain project_path, raw_data_path, fig_path, and dict of group info.

    Args:
        config_path (str): Path to YAML project file

    Raises:
        ex: Error reading config

    Returns:
        YAML object : YAML object containing relevant experiment information.
    """
    try:
        with open(config_path, "r") as file:
            expt_config = yaml.safe_load(file)
    except Exception as ex:
        print("Error reading the config file")
        raise ex

    return expt_config


def update_expt_config(expt_config, config_filename, update_dict):
    """
    Update an expt_config with information provided in update_dict.
    The keys in update_dict should be identical to expt_config.

    Args:
        expt_config (dict): expt_config.yml file
        config_filename (str): Name of expt_config file
        update_dict (dict): dict of keys in expt_config to update.
    """

    if "dirs" in update_dict.keys():
        dir_dict = update_dict.pop("dirs")
        # update params contianing project_path
        if "project" in dir_dict.keys():
            dir_dict["figures"] = dir_dict["project"] + "/figures/"
            dir_dict["data"] = dir_dict["project"] + "/data/"
            dir_dict["fp_data"] = dir_dict["project"] + "/raw/photometry/"
            dir_dict["beh_data"] = dir_dict["project"] + "/raw/behavior"
        for key, val in dir_dict.items():
            expt_config["dirs"][key] = val
    for key in update_dict:
        expt_config[key] = update_dict[key]
        print(f"expt_config[{key}] has been updated.")

    # dump yaml into new file
    with open(config_filename, "w") as f:
        f.write(yaml_dump(expt_config))
    #
    print(f"{config_filename} has been updated!")
