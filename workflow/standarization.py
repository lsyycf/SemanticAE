import argparse
import contextlib
import os
import sys
from datetime import datetime
from enum import Enum

from omegaconf import OmegaConf

from workflow.utils import copy_sources


FRAMEWORKNAME = "deepadv"
"""
Call this before parsing the arguments
初始化标准化配置。

    根据提供的参数和配置文件，初始化项目所需的全局配置。该函数设置命令行参数解析，
    并根据新配置标志决定默认创建新配置还是加载现有配置。

    参数:
    - name (str): 项目名称，默认值为传入的name参数。
    - cfg (str, 可选): 配置文件名称，默认值为None。
    - parser (argparse.ArgumentParser, 可选): 命令行参数解析器，默认值为None。
    - parse_yaml (bool, 可选): 是否使用内置yaml config解析器件，默认值为True。
    - use_new_config_default (bool, 可选): 是否默认使用新配置，默认为True。
    
    Initialize standardized configuration: 
    Based on the provided parameters and configuration files, initialize the global configuration required by the project. 
    This function sets up command-line argument parsing and decides whether to create a new configuration or load an existing 
    one based on the new config flag.

    Parameters:
    - name (str): Project name, defaults to the input name parameter.
    - cfg (str, optional): Configuration file name, defaults to None.
    - parser (argparse.ArgumentParser, optional): Command-line argument parser, defaults to None.
    - parse_yaml (bool, optional): Whether to use built-in YAML config parser component, defaults to True.
    - use_new_config_default (bool, optional): Whether to use new configuration by default, defaults to True.
"""


def init_standardization(name = None, cfg=None, parser: argparse.ArgumentParser = None, parse_yaml=True,
                         use_new_config_default=True):
    if name is None:
        name = os.path.basename(os.path.dirname(os.path.abspath(sys.argv[0])))
    if parser is None:
        parser = argparse.ArgumentParser()
    # 添加命令行参数解析选项
    parser.add_argument("-n", "--project_name", help="Experiment Name", default=name)
    parser.add_argument("-c", "--config", help="Config Name", default=cfg)
    parser.add_argument("-i", "--id", help="Experiment ID", default="base")
    parser.add_argument("-d", "--description", help="Description", default="")
    parser.add_argument("--newconfig", action="store_true", default=use_new_config_default,
                        help="Load a new config and override the existing yaml config.")

    # 解析命令行参数
    args, extras = parser.parse_known_args()
    # erase the leading -- in extras.
    for i, arg in enumerate(extras):
        if arg.startswith("--"):
            extras[i] = arg[2:]

    # 更新全局设置的配置
    GlobalSettings.config = args.config

    GlobalSettings.experiment_id = args.id
    GlobalSettings.project_name = args.project_name
    GlobalSettings.description = args.description

    # 环境配置
    import resource
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))


    if not parse_yaml:
        return

    global global_args_base, cli_args



    # OmegaConf.register_new_resolver("path_datasets", lambda _: GlobalSettings.data_root)
    # OmegaConf.reguster_new_resolver("PATH_LOGS", lambda _: GlobalSettings.log_root)


    # 根据是否需要新配置，决定是创建空配置还是加载现有配置
    if args.newconfig:
        global_args_base = OmegaConf.create()
    else:
        log_config_path = os.path.join(GlobalSettings.get_path(GlobalSettings.PathType.LOG), "config.yaml")
        try:
            global_args_base = OmegaConf.load(log_config_path)
        except FileNotFoundError:
            print(f"Logged config file not found at {log_config_path}. Creating an empty config.")
            global_args_base = OmegaConf.create()

    cli_args = OmegaConf.from_cli(extras)


    # 根据是否需要新配置，决定配置的合并方式
    if args.newconfig:
        config_path = GlobalSettings.get_path(GlobalSettings.PathType.CONFIG)

        try:
            yaml_args = OmegaConf.load(config_path)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Config file not found at {config_path}. Please check the commandline arguments: {args}")

        global_args_base = OmegaConf.merge(yaml_args, cli_args)
    else:
        global_args_base = OmegaConf.merge(global_args_base, cli_args)
    # print(OmegaConf.has_resolver("PATH_DATASET"))
    # Register the resolver for config referencing


    global_args_base.PATH_DATASET = GlobalSettings.data_root.rstrip("/")
    global_args_base.PATH_LOGS = GlobalSettings.log_root.rstrip("/")
    # trim the tailing "/" in GlobalSettings.log_root
    global_args_base.PATH_CONFIGS = GlobalSettings.configs_dir.rstrip("/")
    global_args_base.PATH_CONFIG = GlobalSettings.configs_dir.rstrip("/")
    OmegaConf.register_new_resolver("yaml", lambda path: OmegaConf.load(path)) # note: may have security issues.

    OmegaConf.resolve(global_args_base)
    print(global_args_base)
    return args


global_args_base = OmegaConf.create()
cli_args = OmegaConf.create()

def get_args():
    return global_args_base


def set_global_exp_id(id: str):
    GlobalSettings.experiment_id = id


"""
Call this after all args have been parsed and before the experiment launched.
Save the global config for training restorage
"""


def args_collect_standardization(save_config = True):
    # OmegaConf.merge(global_args, args)
    # global_args.update(dict(
    #     project_name=GlobalSettings.project_name,
    #     config=GlobalSettings.config,
    #     experiment_id=GlobalSettings.experiment_id,
    #     description=GlobalSettings.description
    # ))

    copy_sources(os.path.join(GlobalSettings.sources_dir, GlobalSettings.project_name),
                 os.path.join(GlobalSettings.get_path(GlobalSettings.PathType.LOG), "source_code"))
    if save_config:
        # mkdirs
        os.makedirs(GlobalSettings.get_path(GlobalSettings.PathType.LOG), exist_ok=True)
        OmegaConf.save(global_args_base, os.path.join(GlobalSettings.get_path(GlobalSettings.PathType.LOG), "config.yaml"),
                       resolve=True)
        OmegaConf.save(cli_args, os.path.join(GlobalSettings.get_path(GlobalSettings.PathType.LOG), "config_cli.yaml"),
                       resolve=False)


    # Save Notes to Log Dir. 保存实验记录
    if GlobalSettings.description != "":
        GlobalSettings.save_notes(GlobalSettings.description)


"""
Use this to obtain the data, model and log paths.
Related file structure will be automatically created.
"""
path_to_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
path_cfgs = OmegaConf.load(path_to_root + "/workflow/path_cfg.yml")

class GlobalSettings:
    log_root = os.path.join(path_to_root, path_cfgs.logs_dir)
    sources_dir = os.path.join(path_to_root, "sources_root")
    configs_dir = os.path.join(path_to_root, "configs")
    data_root = os.path.join(path_to_root, path_cfgs.dataset_dir)
    project_name = ""
    config = ""
    experiment_id = ""
    description = ""


    class PathType:
        LOG = 0
        DATA = 1
        GENERATED_DATA = 2
        SOURCES = 3
        CONFIG = 4



    """
    We use different policy for different type of storage.
    Log Location: ProjectName / ConfigName / ExperimentID /
    Origin Data Location: data_root / DataName
    Generated Data Location: data_root / DataName / ProjectName / ConfigName /, and _config.yaml is saved together.
    """

    @staticmethod
    def get_path(type, data_name=None, create=False):
        if type == GlobalSettings.PathType.LOG:
            path = os.path.join(GlobalSettings.log_root, GlobalSettings.project_name, GlobalSettings.config,
                                GlobalSettings.experiment_id)
            if create and not os.path.exists(path):
                os.makedirs(path)
            return path

        elif type == GlobalSettings.PathType.DATA:
            assert data_name is not None
            return os.path.join(GlobalSettings.data_root, data_name, "original")

        elif type == GlobalSettings.PathType.GENERATED_DATA:
            assert data_name is not None
            path = os.path.join(GlobalSettings.data_root, data_name, GlobalSettings.project_name, GlobalSettings.config)
            if create:
                description = path + "_" + "config.yaml"
                OmegaConf.save(global_args_base, description, resolve=True)
                if not os.path.exists(path):
                    os.makedirs(path)
            return path

        elif type == GlobalSettings.PathType.SOURCES:
            return GlobalSettings.sources_dir

        elif type == GlobalSettings.PathType.CONFIG:
            path = os.path.join(GlobalSettings.configs_dir, GlobalSettings.project_name, GlobalSettings.config)
            if not path.endswith(".yaml") or not path.endswith(".yml") or not path.endswith(".json"):
                path = path + ".yaml"
            return path

    @staticmethod
    def save_notes(description):
        now = datetime.now()
        datetime_str = now.strftime("%Y-%m-%d %H:%M:%S")
        description = f"{description} -- {datetime_str}"
        notes = os.path.join(GlobalSettings.get_path(GlobalSettings.PathType.LOG), "exp_notes.txt")
        with open(notes, "a") as f:
            f.write("\n" + description)



@contextlib.contextmanager
def temp_chdir(path):
    """
    A context manager that temporarily changes the current working directory to the given path.
    """
    current_dir = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(current_dir)



if __name__ == '__main__':
    args = init_standardization("test")
    args_collect_standardization()

