import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import argparse
import sys
import os
import re
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

from coil_core import execute_train, execute_validation, execute_drive, folder_execute
from coilutils.general import create_log_folder, create_exp_path, erase_logs,\
                          erase_wrong_plotting_summaries, erase_validations

# TODO: make it clearer what args are for each command, otherwise it's a nightmare as a user; finish the following
_examples = '''examples:
    # Train a model with a configuration file
    python %(prog)s train --gpus 0 --folder ETE --exp ETE_resnet50_1
    # Validate a trained model with the dataset in the configuration file
    python %(prog)s validate --gpus 0 --folder ETE --exp ETE_resnet50_1
    # Drive a trained model
    python %(prog)s drive --gpus 0 
'''


def _parse_num_range(s):
    # string s will be either a range '0-3', so we match with regular expression
    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return range(int(m.group(1)), int(m.group(2)) + 1)
    # Or s will be a comma-separated list of numbers '0,2,3'
    values = s.split(',')
    return [int(x) for x in values]


def main():
    # parser = argparse.ArgumentParser(description=__doc__, epilog=_examples,
    #                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    #
    # subparsers = parser.add_subparsers(help='Sub-commands', dest='command')
    #
    # parser_train = subparsers.add_parser('train', help='Train the model.')
    # parser_train.add_argument('--gpus', type=_parse_num_range)
    # parser_train.set_defaults(func=execute_train)
    #
    # parser_validate = subparsers.add_parser('validate', help='Validate a pretrained model.')
    # parser_validate.set_defaults(func=execute_validation)
    #
    # parser_drive = subparsers.add_parser('drive', help='Drive with a pretrained model.')
    # parser_drive.set_defaults(func=execute_drive)
    #
    # args = parser.parse_args()
    # kwargs = vars(args)
    # subcommand = kwargs.pop('command')
    #
    # if subcommand is None:
    #     print('Error: missing subcommand. Re-run with --help for usage.')
    #     sys.exit(1)
    #
    # func = kwargs.pop('func')
    # func(**kwargs)

    argparser = argparse.ArgumentParser(description=__doc__)

    argparser.add_argument(
        '--single-process',  # to be replaced with subcommands (see above)
        help='Choose between training your model (train), validating (validation), or drive (drive).',
        default='train',
        type=str,
        required=True
    )
    argparser.add_argument(
        '--gpus',  # execute_{train, validation, drive}, folder_execute (hence more than one gpu)
        nargs='+',
        dest='gpus',
        type=str
    )
    argparser.add_argument(
        '-f',
        '--folder',  # execute_{train, validation, drive}, folder_execute
        type=str,
        required=True
    )
    argparser.add_argument(
        '-e',
        '--exp',  # execute_{train, validation, drive}
        type=str
    )
    argparser.add_argument(
        '-vd',
        '--val-datasets',  # folder_execute; args.erase_bad_validations and args.restart_validation
        dest='validation_datasets',
        nargs='+',
        default=[]
    )
    argparser.add_argument(
        '--no-train',
        dest='is_training',  # folder_execute
        action='store_false'
    )
    argparser.add_argument(
        '-de',
        '--drive-envs',  # execute_drive, folder_execute
        dest='driving_environments',
        nargs='+',
        default=[]
    )
    argparser.add_argument(
        '-v', '--verbose',  # not used???
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '-ebv', '--erase-bad-validations',  # Add all of this before the validation code
        action='store_true',
        dest='erase_bad_validations',
        help='erase the bad validations (Incomplete)'
    )
    argparser.add_argument(
        '-rv', '--restart-validations',  # Ibidem
        action='store_true',
        dest='restart_validations',
        help='Set to carla to run offscreen'
    )
    argparser.add_argument(
        '-gv',
        '--gpu-value',
        dest='gpu_value',  # folder_execute
        type=float,
        default=3.5
    )
    argparser.add_argument(
        '-nw',
        '--number-of-workers',
        dest='number_of_workers',  # execute_train, folder_execute
        type=int,
        default=12
    )
    argparser.add_argument(
        '-ns', '--no-screen',
        action='store_true',
        dest='no_screen',  # execute_drive (params)
        help='Set to carla to run offscreen'
    )
    argparser.add_argument(
        '-dk', '--docker',
        dest='docker',  # execute_drive (params)
        default='carlasim/carla:0.8.4',
        type=str,
        help='Set to run carla using docker'
    )
    argparser.add_argument(
        '-rc', '--record-collisions',
        action='store_true',
        dest='record_collisions',  # execute_drive (params)
        help='Set to run carla using docker'
    )
    args = argparser.parse_args()

    # TODO: Combine this into _parse_num_range
    # Check if the list of GPUs is valid (they are all ints)
    for gpu in args.gpus:
        try:
            int(gpu)
        except ValueError:
            raise ValueError(f"GPU {gpu} is not a valid int number")

    # Check if the driving parameters are passed in a correct way
    if args.driving_environments is not None:
        for de in list(args.driving_environments):
            if len(de.split('_')) < 2:
                raise ValueError("Invalid format for the driving environments should be Suite_Town")

    # This is the folder creation of the
    create_log_folder(args.folder)
    erase_logs(args.folder)
    if args.erase_bad_validations:
        erase_wrong_plotting_summaries(args.folder, list(args.validation_datasets))
    if args.restart_validations:
        erase_validations(args.folder, list(args.validation_datasets))

    # There are two modes of execution
    if args.single_process is not None:
        ####
        # MODE 1: Single Process. Just execute a single experiment alias.
        ####

        if args.exp is None:
            raise ValueError(" You should set the exp alias when using single process")

        if args.single_process == 'train':
            execute_train(gpu=args.gpus[0], exp_folder=args.folder, exp_alias=args.exp,
                          suppress_output=False, number_of_workers=args.number_of_workers)

        elif args.single_process == 'validation':
            execute_validation(gpu=args.gpus[0], exp_folder=args.folder, exp_alias=args.exp, suppress_output=False)

        elif args.single_process == 'drive':
            # The definition of parameters for driving
            drive_params = {
                "suppress_output": False,
                "no_screen": args.no_screen,
                "docker": args.docker,
                "record_collisions": args.record_collisions
            }
            execute_drive(gpu=args.gpus[0], exp_folder=args.folder, exp_alias=args.exp,
                          exp_set_name=list(args.driving_environments)[0], params=drive_params)

        else:
            raise Exception("Invalid name for single process, chose from (train, validation, test)")

    else:
        ####
        # MODE 2: Folder execution. Execute train/validation/drive for all experiments on
        #         a certain training folder
        ####
        # We set by default that each gpu has a value of 3.5, allowing a training and
        # a driving/validation
        allocation_parameters = {'gpu_value': args.gpu_value,
                                 'train_cost': 1.5,
                                 'validation_cost': 1.0,
                                 'drive_cost': 1.5}

        params = {
            'folder': args.folder,
            'gpus': list(args.gpus),
            'is_training': args.is_training,
            'validation_datasets': list(args.validation_datasets),
            'driving_environments': list(args.driving_environments),
            'driving_parameters': drive_params,
            'allocation_parameters': allocation_parameters,
            'number_of_workers': args.number_of_workers

        }

        folder_execute(params)
        print("SUCCESSFULLY RAN ALL EXPERIMENTS")


if __name__ == '__main__':
    main()
