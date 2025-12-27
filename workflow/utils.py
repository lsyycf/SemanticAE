import argparse
import os
import shutil

from easydict import EasyDict


def copy_sources(source_parent_dir, output_root):
    for root, dirs, files in os.walk(source_parent_dir, followlinks=True):
        rt = root[len(source_parent_dir) + 1:]
        for file in files:
            if os.path.splitext(file)[1] in [".py", ".yml", ".yaml", "json", ".sh"] and \
                    not file.__contains__("_remote_module_non_scriptable"):  # PyCharm Syn
                path_old = os.path.join(root, file)
                path_new = os.path.join(output_root, rt, file)
                os.makedirs(os.path.dirname(path_new), exist_ok=True)
                shutil.copyfile(path_old, path_new)


import yaml
def dump_yaml(args: EasyDict, path):
    """
       Dump an EasyDict object to a YAML file.

       Parameters:
       ----------
       args : EasyDict
           The EasyDict object to be serialized.
       path : str
           The file path where the YAML file will be saved.

       Raises:
       ------
       IOError
           If there is an error writing to the file.
       """
    try:
        # Convert EasyDict to a regular dictionary
        args_dict = dict(args)
        # Open the file and write the YAML content
        with open(path, 'w') as file:
            yaml.dump(args_dict, file, default_flow_style=False, sort_keys=False)
    except IOError as e:
        raise IOError(f"Error writing to file {path}: {e}")

