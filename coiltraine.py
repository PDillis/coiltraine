import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import argparse
import sys
import os
import re
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

from coil_core import execute_train, execute_validation, execute_drive, folder_execute


# TODO: finish examples
_examples = '''examples:
    # Train a model with a configuration file
    python %(prog)s train --gpus 0 --folder ETE --exp ETE_resnet50_1
    # Validate a trained model with the dataset in the configuration file
    python %(prog)s validate --gpus 0 --folder ETE --exp ETE_resnet50_1
    # Drive a trained model
    python %(prog)s drive --gpus 0 
    # Execute all experiments in a folder
    python %(prog)s 
'''


def _parse_gpus(s):
    """
    Accept either a comma separated list of numbers 'a,b,c' or a range 'a-c' and return as a string of list of ints.
    """
    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        range_ints = list(range(int(m.group(1)), int(m.group(2))+1))
        return str(range_ints)
    vals = s.split(',')
    range_ints = [int(x) for x in vals]
    return str(range_ints)


def _parse_driving_envs(driving_environments):
    # Check if the driving parameters are passed in a correct way
    driving_environments = driving_environments.split(',')
    # Check they are in the correct format
    for driving_env in list(driving_environments):
        if len(driving_env.split('_')) < 2:
            raise ValueError(f'Invalid format for the driving environment {driving_env} (should be Suite_Town)')
    # If only one, then we are executing an individual experiment, otherwise a folder_execute
    if len(driving_environments) == 1:
        return driving_environments[0]

    return driving_environments


def _parse_val_datasets(validation_datasets):
    # TODO: Is this necessary? Can be replaced in the config of each experiment
    assert len(validation_datasets) > 0, 'Validation datasets cannot be empty!'
    return validation_datasets.split(',')


def main():
    parser = argparse.ArgumentParser(description=__doc__, epilog=_examples,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    subparsers = parser.add_subparsers(help='Sub-commands', dest='command')

    parser_train = subparsers.add_parser('train', help='Train the model')
    parser_train.add_argument('--gpus', type=_parse_gpus, help='Select GPUs to use (replaces CUDA_VISIBLE_DEVICES; default: %(default)s)', default='0')
    parser_train.add_argument('-f', '--folder', type=str, help='Folder name in "configs" where experiment config is held at', dest='exp_folder', required=True)
    parser_train.add_argument('-e', '--exp', type=str, help='Experiment config (yaml file) name.', dest='exp_alias', required=True)
    parser_train.add_argument('--suppress-output', type=bool, help='Suppress output to console and save to "_output_logs"', default=False)
    parser_train.add_argument('-nw', '--number-of-workers', type=int, help='Number of threads used for DataLoader', default=12)
    parser_train.set_defaults(func=execute_train)

    parser_validate = subparsers.add_parser('validate', help='Validate a pretrained model')
    parser_validate.add_argument('--gpus', type=_parse_gpus, help='Select GPUs to use (replaces CUDA_VISIBLE_DEVICES; default: %(default)s)', default='0')
    parser_validate.add_argument('-f', '--folder', type=str,  help='Folder name in "configs" where experiment config is held at', dest='exp_folder', required=True)
    parser_validate.add_argument('-e', '--exp', type=str, help='Experiment config (yaml file) name', dest='exp_alias', required=True)
    parser_validate.add_argument('--suppress-output', type=bool, help='Suppress output to console and save to "_output_logs"', default=False)
    parser_validate.add_argument('-ebv', '--erase-bad-validations', action='store_true', help='Erase the bad validations (Incomplete)', dest='erase_bad_validations')
    parser_validate.add_argument('-rv', '--restart-validations', action='store_true', help='Restart validations', dest='restart_validations')
    parser_validate.set_defaults(func=execute_validation)

    parser_drive = subparsers.add_parser('drive', help='Drive with a pretrained model')
    parser_drive.add_argument('--gpus', type=_parse_gpus, help='Select GPUs to use (replaces CUDA_VISIBLE_DEVICES; default: %(default)s)', default='0')
    parser_drive.add_argument('-f', '--folder', type=str,  help='Folder name in "configs" where experiment config is held at', dest='exp_folder', required=True)
    parser_drive.add_argument('-e', '--exp', type=str, help='Experiment config (yaml file) name.', dest='exp_alias', required=True)
    parser_drive.add_argument('-de', '--drive-envs', type=_parse_driving_envs, help='Driving environments where models will be tesested', dest='exp_set_name')
    parser_drive.add_argument('-dk', '--docker', type=str, help='Docker image to run CARLA with (default: %(default)s)', default='carla_0911_t01')
    parser_drive.add_argument('-ns', '--no-screen', action='store_true', help='Set CARLA to run offscreen', dest='no_screen')
    parser_drive.add_argument('-rc', '--record-collisions', action='store_true', help='Record collisions during drive')
    parser_drive.add_argument('--suppress-output', type=bool, help='Suppress output to console and save to "_output_logs"', default=False)
    parser_drive.set_defaults(func=execute_drive)

    # TODO: I won't use this portion of the code, but it's perhaps useful for someone else
    # Folder execution. Execute train/validation/drive for all experiments on a certain training folder
    parser_folder_execute = subparsers.add_parser('folder-execute', help='Execute all experiments in a folder')
    parser_folder_execute.add_argument('--gpus', type=_parse_gpus, help='Select GPUs to use (replaces CUDA_VISIBLE_DEVICES; default: %(default)s)', default='0')
    parser_folder_execute.add_argument('-f', '--folder', type=str, help='Folder name in "configs" where experiment config is held at', dest='exp_folder', required=True)
    parser_folder_execute.add_argument('-vd', '--val-datasets', type=_parse_val_datasets, help='Validation datasets to restart or delete')
    parser_folder_execute.add_argument('-ebv', '--erase-bad-validations', action='store_true',  help='Erase the bad validations (Incomplete)', dest='erase_bad_validations')
    parser_folder_execute.add_argument('-rv', '--restart-validations', action='store_true', help='Restart validations', dest='restart_validations')
    parser_folder_execute.add_argument('--no-train', action='store_false', help="Don't train the experiments in the folder", dest='is_training')
    parser_folder_execute.add_argument('-de', '--drive-envs', type=_parse_driving_envs, help='Driving environments where models will be tesested', dest='exp_set_name')
    parser_folder_execute.add_argument('-nw', '--number-of-workers', type=int, help='Number of threads used for DataLoader', default=12)
    parser_folder_execute.add_argument('-ns', '--no-screen', action='store_true', help='Set CARLA to run offscreen', dest='no_screen')
    parser_folder_execute.add_argument('-dk', '--docker', type=str, help='Docker image to run CARLA with (default: %(default)s)', default='carla_0911_t01')
    parser_folder_execute.add_argument('-rc', '--record-collisions', action='store_true', help='Record collisions during drive')
    parser_folder_execute.set_defaults(func=folder_execute)

    args = parser.parse_args()
    kwargs = vars(args)
    subcommand = kwargs.pop('command')

    if subcommand is None:
        print('Error: missing subcommand. Re-run with --help for usage.')
        sys.exit(1)

    func = kwargs.pop('func')
    func(**kwargs)


if __name__ == '__main__':
    main()
