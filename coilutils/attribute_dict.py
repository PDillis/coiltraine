"""A simple attribute dictionary used for representing configuration options."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from ast import literal_eval
import copy
import numpy as np

_VALID_TYPES = {tuple, list, str, int, float, bool, type(None), np.ndarray, range, dict, set}


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
            value = literal_eval(value)
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
        except (SyntaxError, ValueError):
            pass
        return value

    @classmethod
    def _create_config_tree_from_dict(cls, dic):
        '''
        Create a configuration tree using the given dictionary. Thus, any dict-like objects inside dic
        will be treated as a new AttributeDict
        Args:
            dic (dict): dictionary to create the attribute dict from
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


def _merge_a_into_b(a, b, key_list=[]):
    '''
    Args:
        :a (dict, AttrDict):
        :b (dict, AttrDict):
        :key_list ():
    '''
    assert isinstance(a, AttributeDict), f'Argument `a` must be an AttrDict; cur type: {type(a)}'
    assert isinstance(b, AttributeDict), f'Argument `b` must be an AttrDict; cur type: {type(b)}'

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


def _check_and_coerce_cfg_value_type(replacement, original, key, full_key):
    """Checks that `replacement`, which is intended to replace `original` is of
    the right type. The type is correct if it matches exactly or is one of a few
    cases in which the type can be easily coerced.
    """

    def extract_type(s):
        '''
        Checks and extracts the actual type of the argument s
        Args:
            s: argument to extract the type from
        Example:
            'a' is of type str; 'None' is of type NoneType; 'range(0, 1)' is of type range,
            None is also of type NoneType, np.array(0) is of type np.ndarray, etc.
        '''
        if isinstance(s, str):
            try:
                return type(eval(s))
            except (NameError, SyntaxError):
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
