from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from coilutils import AttributeDict, _merge_a_into_b

import os
import yaml

from configs.namer import generate_name
from logger.coil_logger import create_log, add_message


_g_conf = AttributeDict()
_g_conf.immutable(False)

"""#### GENERAL CONFIGURATION PARAMETERS ####"""
_g_conf.NUMBER_OF_LOADING_WORKERS = 12
_g_conf.FINISH_ON_VALIDATION_STALE = None
_g_conf.EXPERIENCE_FILE = ''

"""#### INPUT RELATED CONFIGURATION PARAMETERS ####"""
_g_conf.SENSORS = {'rgb': (3, 88, 200)}
_g_conf.MEASUREMENTS = {'float_data': (31)}
_g_conf.TARGETS = ['steer', 'throttle', 'brake']
_g_conf.INPUTS = ['speed_module']
_g_conf.INTENTIONS = []
_g_conf.BALANCE_DATA = True
_g_conf.STEERING_DIVISION = [0.05, 0.05, 0.1, 0.3, 0.3, 0.1, 0.05, 0.05]
_g_conf.PEDESTRIAN_PERCENTAGE = 0
_g_conf.SPEED_DIVISION = []
_g_conf.LABELS_DIVISION = [[0, 2, 5], [3], [4]]
_g_conf.BATCH_SIZE = 120
_g_conf.SPLIT = None
_g_conf.REMOVE = None
_g_conf.AUGMENTATION = None


_g_conf.DATA_USED = 'all' #  central, all, sides,
_g_conf.USE_NOISE_DATA = True
_g_conf.TRAIN_DATASET_NAME = '1HoursW1-3-6-8'  # We only set the dataset in configuration for training
_g_conf.LOG_SCALAR_WRITING_FREQUENCY = 2   # TODO NEEDS TO BE TESTED ON THE LOGGING FUNCTION ON  CREATE LOG
_g_conf.LOG_IMAGE_WRITING_FREQUENCY = 1000
_g_conf.EXPERIMENT_BATCH_NAME = "eccv"
_g_conf.EXPERIMENT_NAME = "default"
_g_conf.EXPERIMENT_GENERATED_NAME = None

# TODO: not necessarily the configuration need to know about this
_g_conf.PROCESS_NAME = None
_g_conf.NUMBER_ITERATIONS = 20000
_g_conf.SAVE_SCHEDULE = 'range(0, 2000, 200)'
_g_conf.NUMBER_FRAMES_FUSION = 1
_g_conf.NUMBER_IMAGES_SEQUENCE = 1
_g_conf.SEQUENCE_STRIDE = 1
_g_conf.TEST_SCHEDULE = 'range(0, 2000, 200)'
_g_conf.SPEED_FACTOR = 12.0
_g_conf.AUGMENT_LATERAL_STEERINGS = 6
_g_conf.NUMBER_OF_HOURS = 1
_g_conf.WEATHERS = [1, 3, 6, 8]
#### Starting the model by loading another
_g_conf.PRELOAD_MODEL_BATCH = None
_g_conf.PRELOAD_MODEL_ALIAS = None
_g_conf.PRELOAD_MODEL_CHECKPOINT = None
_g_conf.MODEL_CONFIGURATION = {}

"""#### Network Related Parameters ####"""


_g_conf.MODEL_TYPE = 'coil_icra'
_g_conf.PRE_TRAINED = False
_g_conf.MAGICAL_SEED = 42


_g_conf.LEARNING_RATE_DECAY_INTERVAL = 50000
_g_conf.LEARNING_RATE_DECAY_LEVEL = 0.5
_g_conf.LEARNING_RATE_THRESHOLD = 1000
_g_conf.LEARNING_RATE = 0.0002  # First
_g_conf.BRANCH_LOSS_WEIGHT = [0.95, 0.95, 0.95, 0.95, 0.05]
_g_conf.VARIABLE_WEIGHT = {'Steer': 0.5, 'Gas': 0.45, 'Brake': 0.05}
_g_conf.USED_LAYERS_ATT = []

_g_conf.LOSS_FUNCTION = 'L2'

"""#### Simulation Related Parameters ####"""

_g_conf.IMAGE_CUT = [115, 510]  # How you should cut the input image that is received from the server
_g_conf.USE_ORACLE = False
_g_conf.USE_FULL_ORACLE = False
_g_conf.AVOID_STOPPING = False


def merge_with_yaml(yaml_filename):
    """Load a yaml config file and merge it into the global config object"""
    global _g_conf
    with open(yaml_filename, 'r') as f:
        yaml_file = yaml.load(f, Loader=yaml.BaseLoader)
        yaml_cfg = AttributeDict(yaml_file)

    _merge_a_into_b(yaml_cfg, _g_conf)

    path_parts = os.path.split(yaml_filename)
    _g_conf.EXPERIMENT_BATCH_NAME = os.path.split(path_parts[-2])[-1]
    _g_conf.EXPERIMENT_NAME = path_parts[-1].split('.')[-2]
    _g_conf.EXPERIMENT_GENERATED_NAME = generate_name(_g_conf)


def get_names(folder):
    alias_in_folder = os.listdir(os.path.join('configs', folder))

    experiments_in_folder = {}
    for experiment_alias in alias_in_folder:

        g_conf.immutable(False)
        merge_with_yaml(os.path.join('configs', folder, experiment_alias))

        experiments_in_folder.update({experiment_alias: g_conf.EXPERIMENT_GENERATED_NAME})

    return experiments_in_folder


# TODO: Make this nicer, now it receives only one parameter
def set_type_of_process(process_type, param=None):
    """
    This function is used to set which is the type of the current process, test, train or val
    and also the details of each since there could be many vals and tests for a single
    experiment.

    NOTE: AFTER CALLING THIS FUNCTION, THE CONFIGURATION CLOSES

    Args:
        type:

    Returns:

    """

    if _g_conf.PROCESS_NAME == "default":
        raise RuntimeError(" You should merge with some exp file before setting the type")

    if process_type == 'train':
        _g_conf.PROCESS_NAME = process_type
    elif process_type == "validation":
        _g_conf.PROCESS_NAME = process_type + '_' + param
    if process_type == "drive":  # FOR drive param is city name.
        _g_conf.CITY_NAME = param.split('_')[-1]
        _g_conf.PROCESS_NAME = process_type + '_' + param

    #else:  # FOr the test case we join with the name of the experimental suite.

    create_log(_g_conf.EXPERIMENT_BATCH_NAME,
               _g_conf.EXPERIMENT_NAME,
               _g_conf.PROCESS_NAME,
               _g_conf.LOG_SCALAR_WRITING_FREQUENCY,
               _g_conf.LOG_IMAGE_WRITING_FREQUENCY)

    if process_type == "train":
        if not os.path.exists(os.path.join('_logs', _g_conf.EXPERIMENT_BATCH_NAME,
                                            _g_conf.EXPERIMENT_NAME,
                                            'checkpoints') ):
                os.mkdir(os.path.join('_logs', _g_conf.EXPERIMENT_BATCH_NAME,
                                      _g_conf.EXPERIMENT_NAME,
                                      'checkpoints'))

    if process_type == "validation" or process_type == 'drive':
        if not os.path.exists(os.path.join('_logs', _g_conf.EXPERIMENT_BATCH_NAME,
                                           _g_conf.EXPERIMENT_NAME,
                                           _g_conf.PROCESS_NAME + '_csv')):
            os.mkdir(os.path.join('_logs', _g_conf.EXPERIMENT_BATCH_NAME,
                                          _g_conf.EXPERIMENT_NAME,
                                           _g_conf.PROCESS_NAME + '_csv'))


    # TODO: check if there is some integrity.

    add_message('Loading', {'ProcessName': _g_conf.EXPERIMENT_GENERATED_NAME,
                            'FullConfiguration': _g_conf.TRAIN_DATASET_NAME + 'dict'})

    _g_conf.immutable(True)


g_conf = _g_conf

