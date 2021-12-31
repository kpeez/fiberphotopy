"""Code for making expt_config files to describe experimental info."""
import os
import yaml
from ruamel.yaml.main import round_trip_load as yaml_load, round_trip_dump as yaml_dump


def create_expt_config(project_name, project_path, config_name=None):
    """
    Setup and create experiment configuration files.

    Args:
        project_name (str): Project name stored in expt_config["project_name"]
        project_path (str): Path to project
        config_name (str, optional): Prefix for expt_config file. Defaults to None.

    Returns:
        YAML config file named "expt_config-{config_name}.yml".
    """
    # make config file name
    config_filename = "expt_config-" + config_name if config_name else "expt_config"
    config_filename = config_filename + ".yaml"
    # raise error if expt_config already exists
    assert config_filename not in os.listdir(
        project_path
    ), f"{config_filename} already exists"
    old_cwd = os.getcwd()
    if not os.path.exists(project_path):
        os.mkdir(project_path)
    os.chdir(project_path)
    # make default directories
    for folder in ["figures", "data-raw", "data-proc"]:
        if not os.path.exists(str(project_path + folder)):
            os.mkdir(folder)
    # reset working directory
    os.chdir(old_cwd)
    # config template
    configDict = {
        "experiment": project_name,
        "project_path": project_path,
        "fig_path": project_path + "figures/",
        "raw_data_path": project_path + "data-raw/",
        "proc_data_path": project_path + "data-proc/",
        "sessions": {},
        "group_ids": {"group1": list(), "group2": list()},
        "sex_ids": None,
    }
    # lazy convert dict with ruamel
    configDict = yaml_load(yaml_dump(configDict), preserve_quotes=True)
    # add comments to config
    configDict.yaml_set_comment_before_after_key(
        key="project_path", before="Set project and fig paths"
    )
    configDict.yaml_set_comment_before_after_key(key="sessions", before="Session info")
    configDict.yaml_set_comment_before_after_key(key="group_ids", before="Group info")
    # dump yaml into new file
    with open(f"{project_path}{config_filename}", "w") as f:
        f.write(yaml_dump(configDict))

    print(f"{config_filename} has been created in '{project_path}' ")


def update_expt_config(expt_config, config_filename, update_dict):
    """
    Update an expt_config with information provided in update_dict. The keys
    in update_dict should be identical to expt_config.

    Args:
        expt_config (dict): expt_config.yml file
        config_filename (str): Name of expt_config file
        update_dict (dict): dict of keys in expt_config to update.
    """
    if "project_path" in update_dict.keys():
        update_dict["fig_path"] = update_dict["project_path"] + "figures/"
        update_dict["raw_data_path"] = update_dict["project_path"] + "data-raw/"
        update_dict["proc_data_path"] = update_dict["project_path"] + "data-proc/"
    # update expt_config
    for key in update_dict:
        expt_config[key] = update_dict[key]
        print(f"expt_config[{key}] has been updated.")

    # dump yaml into new file
    with open(f"{expt_config['project_path']}{config_filename}.yaml", "w") as f:
        f.write(yaml_dump(expt_config))
    #
    print(f"{config_filename} has been updated!")


def load_expt_config(config_path):
    """
    Load expt_info.yaml to obtain project_path, raw_data_path, fig_path,
    and dict containing any group info.

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
