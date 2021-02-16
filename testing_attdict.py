import copy
import io
import logging
import os
import sys
from ast import literal_eval
import numpy as np
import yaml


# CfgNodes can only contain a limited set of valid types
_VALID_TYPES = {tuple, list, str, int, float, bool, type(None), np.ndarray, range, dict, set}

def _check_and_coerce_cfg_value_type(replacement, original, key, full_key):
    """Checks that `replacement`, which is intended to replace `original` is of
    the right type. The type is correct if it matches exactly or is one of a few
    cases in which the type can be easily coerced.
    """
    def extract_type(s):
        '''
        If
        Args:
            s: argument to extract the type from
        Example:
            'a' is of type str; 'None' is of type NoneType; 'range(0, 1)' is of type range,
            None is also of type NoneType, np.array(0) is of type np.ndarray, etc.
        '''
        if isinstance(s, str):
            try:
                return type(eval(s))
            except NameError:
                return str
        else:
            return type(s)

    original_type = extract_type(original)
    replacement_type = extract_type(replacement)

    # The types must match (with some exceptions)
    if replacement_type == original_type:
        return replacement

    # If either of them is None, allow type conversion to one of the valid types
    if (isinstance(replacement_type, type(None)) and original_type in _VALID_TYPES) or (
        isinstance(original_type, type(None)) and replacement_type in _VALID_TYPES
    ):
        return replacement

    # Cast replacement from from_type to to_type if the replacement and original
    # types match from_type and to_type
    def conditional_cast(from_type, to_type):
        if replacement_type == from_type and original_type == to_type:
            return True, to_type(replacement)
        else:
            return False, None

    # Conditionally casts
    # list <-> tuple
    casts = [(tuple, list), (list, tuple), (dict, AttributeDict), (AttributeDict, dict)]
    # For py2: allow converting from str (bytes) to a unicode string
    try:
        casts.append((str, unicode))  # noqa: F821
    except Exception:
        pass

    for (from_type, to_type) in casts:
        converted, converted_value = conditional_cast(from_type, to_type)
        if converted:
            return converted_value

    raise ValueError(
        "Type mismatch ({} vs. {}) with values ({} vs. {}) for config "
        "key: {}".format(
            original_type, replacement_type, original, replacement, full_key
        )
    )




# ===========================================================================


# def _check_and_coerce_cfg_value_type(value_a, value_b, key, full_key):
#     """Checks that `value_a`, which is intended to replace `value_b` is of the
#     right type. The type is correct if it matches exactly or is one of a few
#     cases in which the type can be easily coerced.
#     """
#     # The types must match (with some exceptions)
#     type_b = type(value_b)
#     type_a = type(value_a)
#     # If they match, we're done
#     if type_a is type_b:
#         return value_a
#     # If one of them is of NoneType, we allow
#     # Exceptions: numpy arrays, strings, tuple<->list
#     if isinstance(value_b, type(None)):
#         value_a = value_a
#     elif isinstance(value_b, np.ndarray):
#         value_a = np.array(value_a, dtype=value_b.dtype)
#     elif isinstance(value_b, str):
#         value_a = str(value_a)
#     elif isinstance(value_a, tuple) and isinstance(value_b, list):
#         value_a = list(value_a)
#     elif isinstance(value_a, list) and isinstance(value_b, tuple):
#         value_a = tuple(value_a)
#     elif isinstance(value_b, range) and not isinstance(value_a, list):
#         value_a = eval(value_a)
#     elif isinstance(value_b, range) and isinstance(value_a, list):
#         value_a = list(value_a)
#     elif isinstance(value_b, dict):
#         value_a = eval(value_a)
#     else:
#         raise ValueError(
#             f'Type mismatch ({type_b} vs. {type_a}) with values ({value_b} vs. {value_a}) for config key: {full_key}'
#         )
#     return value_a


yaml_cfg = {'SAVE_SCHEDULE': 'range(0, 500001, 20000)', 'NUMBER_OF_LOADING_WORKERS': 12, 'MAGICAL_SEED': 26957017,
                    'SENSORS': {'rgb_central': [3, 88, 200]}, 'MEASUREMENTS': {'float_data': [31]}, 'BATCH_SIZE': 120,
                    'NUMBER_ITERATIONS': 500001, 'TARGETS': ['steer', 'throttle', 'brake'], 'INPUTS': ['forward_speed'],
                    'NUMBER_FRAMES_FUSION': 1, 'NUMBER_IMAGES_SEQUENCE': 1, 'SEQUENCE_STRIDE': 1, 'AUGMENT_LATERAL_STEERINGS': 6,
                    'SPEED_FACTOR': 12.0, 'TRAIN_DATASET_NAME': 'sample_dataset', 'AUGMENTATION': 'None', 'DATA_USED': 'all',
                    'USE_NOISE_DATA': True, 'NUMBER_OF_HOURS': 1,
                    'EXPERIENCE_FILE': '/home/dporres/Documents/code/cexp/database/sample_dataset.json',
                    'TEST_SCHEDULE': 'range(100000, 100001, 100000)', 'MODEL_TYPE': 'coil-icra',
                    'MODEL_CONFIGURATION': {'perception': {'res': {'name': 'resnet34', 'num_classes': 512}},
                                            'measurements': {'fc': {'neurons': [128, 128], 'dropouts': [0.0, 0.0]}},
                                            'join': {'fc': {'neurons': [512], 'dropouts': [0.0]}},
                                            'speed_branch': {'fc': {'neurons': [256, 256], 'dropouts': [0.0, 0.5]}},
                                            'branches': {'number_of_branches': 4,
                                                         'fc': {'neurons': [256, 256], 'dropouts': [0.0, 0.5]}}},
                    'LEARNING_RATE': 0.0002, 'LEARNING_RATE_DECAY_INTERVAL': 75000, 'LEARNING_RATE_THRESHOLD': 5000,
                    'LEARNING_RATE_DECAY_LEVEL': 0.1, 'BRANCH_LOSS_WEIGHT': [0.95, 0.95, 0.95, 0.95, 0.05],
                    'LOSS_FUNCTION': 'L1', 'VARIABLE_WEIGHT': {'Steer': 0.5, 'Gas': 0.45, 'Brake': 0.05},
                    'IMAGE_CUT': [65, 460], 'USE_ORACLE': False, 'USE_FULL_ORACLE': False, 'AVOID_STOPPING': False}
g_conf = {'NUMBER_OF_LOADING_WORKERS': 12, 'FINISH_ON_VALIDATION_STALE': None, 'SENSORS': {'rgb_central': (3, 88, 200)},
                  'MEASUREMENTS': {'float_data': [31]}, 'TARGETS': ['steer', 'throttle', 'brake'], 'INPUTS': ['speed_module'],
                  'INTENTIONS': [], 'BALANCE_DATA': True, 'STEERING_DIVISION': [0.05, 0.05, 0.1, 0.3, 0.3, 0.1, 0.05, 0.05],
                  'PEDESTRIAN_PERCENTAGE': 0, 'SPEED_DIVISION': [], 'LABELS_DIVISION': [[0, 2, 5], [3], [4]],
                  'BATCH_SIZE': 120, 'SPLIT': None, 'REMOVE': None, 'AUGMENTATION': None, 'DATA_USED': 'all',
                  'USE_NOISE_DATA': True, 'TRAIN_DATASET_NAME': '1HoursW1-3-6-8', 'LOG_SCALAR_WRITING_FREQUENCY': 2,
                  'LOG_IMAGE_WRITING_FREQUENCY': 1000, 'EXPERIMENT_BATCH_NAME': 'eccv', 'EXPERIMENT_NAME': 'default',
                  'EXPERIMENT_GENERATED_NAME': None, 'PROCESS_NAME': 'None', 'NUMBER_ITERATIONS': 20000,
                  'SAVE_SCHEDULE': 'range(0, 2000, 200)', 'NUMBER_FRAMES_FUSION': 1, 'NUMBER_IMAGES_SEQUENCE': 1,
                  'SEQUENCE_STRIDE': 1, 'TEST_SCHEDULE': 'range(0, 2000, 200)', 'SPEED_FACTOR': 12.0,
                  'AUGMENT_LATERAL_STEERINGS': 6, 'NUMBER_OF_HOURS': 1, 'WEATHERS': [1, 3, 6, 8],
                  'PRELOAD_MODEL_BATCH': None, 'PRELOAD_MODEL_ALIAS': None, 'PRELOAD_MODEL_CHECKPOINT': None,
                  'MODEL_TYPE': 'coil_icra',
                  'PRE_TRAINED': False, 'MAGICAL_SEED': 42, 'LEARNING_RATE_DECAY_INTERVAL': 50000,
                  'LEARNING_RATE_DECAY_LEVEL': 0.5, 'LEARNING_RATE_THRESHOLD': 1000, 'LEARNING_RATE': 0.0002,
                  'BRANCH_LOSS_WEIGHT': [0.95, 0.95, 0.95, 0.95, 0.05],
                  'USED_LAYERS_ATT': [], 'LOSS_FUNCTION': 'L2', 'IMAGE_CUT': [115, 510], 'USE_ORACLE': False,
                  'USE_FULL_ORACLE': False, 'AVOID_STOPPING': False, 'EXPERIENCE_FILE': None}


class AttributeDict(dict):

    IMMUTABLE = '__immutable__'
    NEW_ALLOWED = '__new_allowed__'

    def __init__(self, init_dict=None, new_allowed=False):
        init_dict = {} if init_dict is None else init_dict
        init_dict = self._create_config_tree_from_dict(init_dict)
        super(AttributeDict, self).__init__(init_dict)
        self.__dict__[AttributeDict.IMMUTABLE] = False
        self.__dict__[AttributeDict.NEW_ALLOWED] = new_allowed

    def __getattr__(self, name):
        if name in self.__dict__:
            return self.__dict__[name]
        elif name in self:
            return self[name]
        else:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        if not self.__dict__[AttributeDict.IMMUTABLE]:
            if name in self.__dict__:
                self.__dict__[name] = value
            else:
                self[name] = value
        else:
            raise AttributeError(f'Attempted to set "{name}" to "{value}", but AttributeDict is immutable')

    @classmethod
    def _create_config_tree_from_dict(cls, dic):
        '''
        Create a configuration tree using the given dictionary. Thus, any dict-like objects inside dic
        will be treated as a new AttributeDict
        Args:
            dic (dict):
            key_list (list[str]): list of names which index this AttributeDict from the root; for logging.
        '''
        dic = copy.deepcopy(dic)
        for key, value in dic.items():
            if isinstance(value, dict):
                # Convert dictionary to an AttributeDict
                dic[key] = cls(value)
            else:
                # Check for valid leaf type or nested AttributeDict
                if type(value) not in _VALID_TYPES or isinstance(value, AttributeDict):
                    raise AttributeError(f'Key {key} is not a valid type: {type(value)}.')

        return dic

    def immutable(self, is_immutable):
        """Set immutability to is_immutable and recursively apply the setting
        to all nested AttributeDict.
        """
        self.__dict__[AttributeDict.IMMUTABLE] = is_immutable
        # Recursively set immutable state
        for v in self.__dict__.values():
            if isinstance(v, AttributeDict):
                v.immutable(is_immutable)
        for v in self.values():
            if isinstance(v, AttributeDict):
                v.immutable(is_immutable)

    def is_immutable(self):
        return self.__dict__[AttributeDict.IMMUTABLE]

    def is_new_allowed(self):
        return self.__dict__[AttributeDict.NEW_ALLOWED]

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, super(AttributeDict, self).__repr__())

    @classmethod
    def _decode_cfg_value(cls, value):
        """Decodes a raw config value (e.g., from a yaml config files or command
        line argument) into a Python object.
        """
        # Configs parsed from raw yaml will contain dictionary keys that need to be
        # converted to AttrDict objects
        if isinstance(value, dict):
            return cls(value)
        # All remaining processing is only applied to strings
        if not isinstance(value, str):
            return value
        # Try to interpret `v` as a:
        #   string, number, tuple, list, dict, boolean, or None
        try:
            v = literal_eval(value)
        # The following two excepts allow v to pass through when it represents a
        # string.
        # Longer explanation:
        # The type of v is always a string (before calling literal_eval), but
        # sometimes it *represents* a string and other times a data structure, like
        # a list. In the case that v represents a string, what we got back from the
        # yaml parser is 'foo' *without quotes* (so, not '"foo"'). literal_eval is
        # ok with '"foo"', but will raise a ValueError if given 'foo'. In other
        # cases, like paths (v = 'foo/bar' and not v = '"foo/bar"'), literal_eval
        # will raise a SyntaxError.
        except ValueError:
            pass
        except SyntaxError:
            pass
        return value


def _merge_a_into_b(a, b, key_list=[]):

    assert isinstance(a, AttributeDict) or isinstance(a, dict), f'Argument `a` must be an AttrDict; cur type: {type(a)}'
    assert isinstance(b, AttributeDict) or isinstance(a, dict), f'Argument `b` must be an AttrDict; cur type: {type(b)}'

    for key, value_ in a.items():
        full_key = '.'.join(key_list + [key])

        value = copy.deepcopy(value_)
        value = b._decode_cfg_value(value)

        if key in b:
            value = _check_and_coerce_cfg_value_type(value, b[key], key, full_key)
            # Recursively merge dicts
            if isinstance(value, AttributeDict):
                try:
                    _merge_a_into_b(value, b[key], key_list + [key])
                except BaseException:
                    raise

            else:
                b[key] = value

        elif b.is_new_allowed():
            b[key] = value
        else:
            raise KeyError(f'Non-existent config key: {full_key}')


yaml_cfg = AttributeDict(yaml_cfg)
g_conf = AttributeDict(g_conf, new_allowed=True)

print('yaml config: \n', yaml_cfg, '\n')
print('g config: \n', g_conf, '\n')

_merge_a_into_b(yaml_cfg, g_conf)
print('After merge, g_conf: \n', g_conf)

# test_a = AttributeDict({'a': 5, 'b': {'c': 8}})
# test_b = AttributeDict({'d': 50, 'b': {'c': 20}}, new_allowed=True)
#
# print(test_a)
# print(test_a.a)
# print(test_a.b.c, '\n')
#
# print(test_b)
#
# _merge_a_into_b(test_a, test_b, key_list=[])
#
# print(test_b)

yaml_before = AttributeDict(
    {'SAVE_SCHEDULE': 'range(0, 500001, 20000)',
     'NUMBER_OF_LOADING_WORKERS': 12,
     'MAGICAL_SEED': 26957017,
     'SENSORS': AttributeDict({'rgb': [3, 88, 200]}),
     'MEASUREMENTS': AttributeDict({'float_data': [31]}),
     'BATCH_SIZE': 120,
     'NUMBER_ITERATIONS': 500001,
     'TARGETS': ['steer', 'throttle', 'brake'],
     'INPUTS': ['forward_speed'],
     'NUMBER_FRAMES_FUSION': 1,
     'NUMBER_IMAGES_SEQUENCE': 1,
     'SEQUENCE_STRIDE': 1,
     'AUGMENT_LATERAL_STEERINGS': 6,
     'SPEED_FACTOR': 12.0,
     'TRAIN_DATASET_NAME': 'sample_dataset',
     'AUGMENTATION': 'None',
     'DATA_USED': 'all',
     'USE_NOISE_DATA': True,
     'NUMBER_OF_HOURS': 1,
     'EXPERIENCE_FILE': '/home/dporres/Documents/code/cexp/database/sample_dataset.json',
     'TEST_SCHEDULE': 'range(100000, 100001, 100000)',
     'MODEL_TYPE': 'coil-icra',
     'MODEL_CONFIGURATION': AttributeDict(
         {'perception': AttributeDict({'res': AttributeDict({'name': 'resnet34', 'num_classes': 512})}),
          'measurements': AttributeDict({'fc': AttributeDict({'neurons': [128, 128], 'dropouts': [0.0, 0.0]})}),
          'join': AttributeDict({'fc': AttributeDict({'neurons': [512], 'dropouts': [0.0]})}),
          'speed_branch': AttributeDict({'fc': AttributeDict({'neurons': [256, 256], 'dropouts': [0.0, 0.5]})}),
          'branches': AttributeDict({'number_of_branches': 4, 'fc': AttributeDict({'neurons': [256, 256], 'dropouts': [0.0, 0.5]})})}),
     'LEARNING_RATE': 0.0002,
     'LEARNING_RATE_DECAY_INTERVAL': 75000,
     'LEARNING_RATE_THRESHOLD': 5000,
     'LEARNING_RATE_DECAY_LEVEL': 0.1,
     'BRANCH_LOSS_WEIGHT': [0.95, 0.95, 0.95, 0.95, 0.05],
     'LOSS_FUNCTION': 'L1',
     'VARIABLE_WEIGHT': AttributeDict({'Steer': 0.5, 'Gas': 0.45, 'Brake': 0.05}),
     'IMAGE_CUT': [65, 460],
     'USE_ORACLE': False,
     'USE_FULL_ORACLE': False,
     'AVOID_STOPPING': False})

g_before = AttributeDict(
    {'NUMBER_OF_LOADING_WORKERS': 12,
     'FINISH_ON_VALIDATION_STALE': None,
     'EXPERIENCE_FILE': '',
     'SENSORS': {'rgb': (3, 88, 200)},
     'MEASUREMENTS': {'float_data': 31},
     'TARGETS': ['steer', 'throttle', 'brake'],
     'INPUTS': ['speed_module'],
     'INTENTIONS': [],
     'BALANCE_DATA': True,
     'STEERING_DIVISION': [0.05, 0.05, 0.1, 0.3, 0.3, 0.1, 0.05, 0.05],
     'PEDESTRIAN_PERCENTAGE': 0,
     'SPEED_DIVISION': [],
     'LABELS_DIVISION': [[0, 2, 5], [3], [4]],
     'BATCH_SIZE': 120,
     'SPLIT': None,
     'REMOVE': None,
     'AUGMENTATION': None,
     'DATA_USED': 'all',
     'USE_NOISE_DATA': True,
     'TRAIN_DATASET_NAME': '1HoursW1-3-6-8',
     'LOG_SCALAR_WRITING_FREQUENCY': 2,
     'LOG_IMAGE_WRITING_FREQUENCY': 1000,
     'EXPERIMENT_BATCH_NAME': 'eccv',
     'EXPERIMENT_NAME': 'default',
     'EXPERIMENT_GENERATED_NAME': None,
     'PROCESS_NAME': 'None',
     'NUMBER_ITERATIONS': 20000,
     'SAVE_SCHEDULE': 'range(0, 2000, 200)',
     'NUMBER_FRAMES_FUSION': 1,
     'NUMBER_IMAGES_SEQUENCE': 1,
     'SEQUENCE_STRIDE': 1,
     'TEST_SCHEDULE': 'range(0, 2000, 200)',
     'SPEED_FACTOR': 12.0,
     'AUGMENT_LATERAL_STEERINGS': 6,
     'NUMBER_OF_HOURS': 1,
     'WEATHERS': [1, 3, 6, 8],
     'PRELOAD_MODEL_BATCH': None,
     'PRELOAD_MODEL_ALIAS': None,
     'PRELOAD_MODEL_CHECKPOINT': None,
     'MODEL_CONFIGURATION': {},
     'MODEL_TYPE': 'coil_icra',
     'PRE_TRAINED': False,
     'MAGICAL_SEED': 42,
     'LEARNING_RATE_DECAY_INTERVAL': 50000,
     'LEARNING_RATE_DECAY_LEVEL': 0.5,
     'LEARNING_RATE_THRESHOLD': 1000,
     'LEARNING_RATE': 0.0002,
     'BRANCH_LOSS_WEIGHT': [0.95, 0.95, 0.95, 0.95, 0.05],
     'VARIABLE_WEIGHT': {},
     'USED_LAYERS_ATT': [],
     'LOSS_FUNCTION': 'L2',
     'IMAGE_CUT': [115, 510],
     'USE_ORACLE': False,
     'USE_FULL_ORACLE': False,
     'AVOID_STOPPING': False})

g_after = AttributeDict(
    {'NUMBER_OF_LOADING_WORKERS': 12,
     'FINISH_ON_VALIDATION_STALE': None,
     'EXPERIENCE_FILE': '/home/dporres/Documents/code/cexp/database/sample_dataset.json',
     'SENSORS': {'rgb': [3, 88, 200]},
     'MEASUREMENTS': {'float_data': [31]},
     'TARGETS': ['steer', 'throttle', 'brake'],
     'INPUTS': ['forward_speed'],
     'INTENTIONS': [],
     'BALANCE_DATA': True,
     'STEERING_DIVISION': [0.05, 0.05, 0.1, 0.3, 0.3, 0.1, 0.05, 0.05],
     'PEDESTRIAN_PERCENTAGE': 0,
     'SPEED_DIVISION': [],
     'LABELS_DIVISION': [[0, 2, 5], [3], [4]],
     'BATCH_SIZE': 120,
     'SPLIT': None,
     'REMOVE': None,
     'AUGMENTATION': 'None',
     'DATA_USED': 'all',
     'USE_NOISE_DATA': True,
     'TRAIN_DATASET_NAME': 'sample_dataset',
     'LOG_SCALAR_WRITING_FREQUENCY': 2,
     'LOG_IMAGE_WRITING_FREQUENCY': 1000,
     'EXPERIMENT_BATCH_NAME': 'eccv',
     'EXPERIMENT_NAME': 'default',
     'EXPERIMENT_GENERATED_NAME': None,
     'PROCESS_NAME': 'None',
     'NUMBER_ITERATIONS': 500001,
     'SAVE_SCHEDULE': range(0, 500001, 20000),
     'NUMBER_FRAMES_FUSION': 1,
     'NUMBER_IMAGES_SEQUENCE': 1,
     'SEQUENCE_STRIDE': 1,
     'TEST_SCHEDULE': range(100000, 100001, 100000),
     'SPEED_FACTOR': 12.0,
     'AUGMENT_LATERAL_STEERINGS': 6,
     'NUMBER_OF_HOURS': 1,
     'WEATHERS': [1, 3, 6, 8],
     'PRELOAD_MODEL_BATCH': None,
     'PRELOAD_MODEL_ALIAS': None,
     'PRELOAD_MODEL_CHECKPOINT': None,
     'MODEL_CONFIGURATION': {
         'perception': AttributeDict({'res': AttributeDict({'name': 'resnet34', 'num_classes': 512})}),
         'measurements': AttributeDict({'fc': AttributeDict({'neurons': [128, 128], 'dropouts': [0.0, 0.0]})}),
         'join': AttributeDict({'fc': AttributeDict({'neurons': [512], 'dropouts': [0.0]})}),
         'speed_branch': AttributeDict({'fc': AttributeDict({'neurons': [256, 256], 'dropouts': [0.0, 0.5]})}),
         'branches': AttributeDict({
             'number_of_branches': 4,
             'fc': AttributeDict({'neurons': [256, 256], 'dropouts': [0.0, 0.5]})})},
     'MODEL_TYPE': 'coil-icra',
     'PRE_TRAINED': False,
     'MAGICAL_SEED': 26957017,
     'LEARNING_RATE_DECAY_INTERVAL': 75000,
     'LEARNING_RATE_DECAY_LEVEL': 0.1,
     'LEARNING_RATE_THRESHOLD': 5000,
     'LEARNING_RATE': 0.0002,
     'BRANCH_LOSS_WEIGHT': [0.95, 0.95, 0.95, 0.95, 0.05],
     'VARIABLE_WEIGHT': {'Steer': 0.5, 'Gas': 0.45, 'Brake': 0.05},
     'USED_LAYERS_ATT': [],
     'LOSS_FUNCTION': 'L1',
     'IMAGE_CUT': [65, 460],
     'USE_ORACLE': False,
     'USE_FULL_ORACLE': False,
     'AVOID_STOPPING': False})


