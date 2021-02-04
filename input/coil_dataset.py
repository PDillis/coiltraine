import os
import glob
import traceback
import collections
import sys
import math
import copy
import json
import random
import numpy as np

from tqdm import tqdm

import torch
import cv2

from torch.utils.data import Dataset

from . import splitter
from . import data_parser

# from cexp.cexp import CEXP
# from cexp.env.scenario_identification import identify_scenario
# from cexp.env.environment import NoDataGenerated

# TODO: Warning, maybe this does not need to be included everywhere.
from configs import g_conf

from coilutils.general import sort_nicely


def join_classes(labels_image, classes_join):
    compressed_labels_image = np.copy(labels_image)
    for key, value in classes_join.items():
        compressed_labels_image[np.where(labels_image == int(key))] = value

    return compressed_labels_image


def parse_remove_configuration(configuration):
    """
    Turns the configuration line of splitting into a name and set of params.
    """
    if configuration is None:
        return 'None', None
    print(f'Configuration: {configuration}')
    config_dict = collections.OrderedDict(configuration)

    name = 'remove'
    for key in config_dict.keys():
        if key != 'weights' and key != 'boost':
            name += f'_{key}'

    return name, config_dict


# def convert_scenario_name_number(measurements):
#     scenario = identify_scenario(measurements['distance_intersection'],
#                                  measurements['road_angle'])
#     if scenario == 'S0_lane_following':
#         return [1, 0, 0, 0]
#     elif scenario == 'S1_lane_following_curve':
#         return [0, 1, 0, 0]
#     elif scenario == 'S2_before_intersection':
#         return [0, 0, 1, 0]
#     elif scenario == 'S3_intersection':
#         return [0, 0, 0, 1]
#     else:
#         raise ValueError(f'Unexpected scenario identified: {scenario}')


def encode_directions(directions):
    if directions == 2.0:
        return [1, 0, 0, 0]
    elif directions == 3.0:
        return [0, 1, 0, 0]
    elif directions == 4.0:
        return [0, 0, 1, 0]
    elif directions == 5.0:
        return [0, 0, 0, 1]
    else:
        raise ValueError(f'Unexpected direction identified: {directions}')


def check_size(img_filename, size):
    img = cv2.imread(img_filename, cv2.IMREAD_COLOR)
    return img.shape[0] == size[1] and img.shape[1] == size[2]


def get_episode_weather(episode):
    with open(os.path.join(episode, 'metadata.json')) as f:
        metadata = json.load(f)
    print(" WEATHER OF EPISODE ", metadata['weather'])
    return int(metadata['weather'])


class CoILDataset(Dataset):
    """ The conditional imitation learning dataset"""

    def __init__(self, root_dir, transform=None, preload_name=None):
        # Setting the root directory for this dataset
        self.root_dir = root_dir
        # We add to the preload name all the remove labels
        if g_conf.REMOVE is not None and g_conf.REMOVE is not "None":
            name, self._remove_params = parse_remove_configuration(g_conf.REMOVE)
            self.preload_name = f'{preload_name}_{name}'
            self._check_remove_function = getattr(splitter, name)
        else:
            self._check_remove_function = lambda _, __: False
            self._remove_params = []
            self.preload_name = preload_name

        preload_path = os.path.join('_preloads', f'{self.preload_name}.npy')
        print(f"Preload name: {self.preload_name}")
        if self.preload_name is not None and os.path.exists(preload_path):
            print(f"Loading from NPY: {preload_path}")
            self.sensor_data_names, self.measurements = np.load(preload_path, allow_pickle=True)
            for key in self.sensor_data_names.keys():
                print(f'\t======> {key} images: {len(self.sensor_data_names[key])}')
            print(f'\t======> measurements: {len(self.measurements)}')
        else:
            print('Creating NPY preload...')
            self.sensor_data_names, self.measurements = self._pre_load_image_folders(root_dir)
            for key in self.sensor_data_names.keys():
                print(f'\t======> {key} images: {len(self.sensor_data_names[key])}')
            print(f'\t======> measurements: {len(self.measurements)}')

        self.transform = transform
        self.batch_read_number = 0

    def __len__(self):
        return len(self.measurements)

    def __getitem__(self, index):
        """
        Get item function used by the dataset loader
        returns all the measurements with the desired image.

        Args:
            index:

        Returns:

        """
        try:
            measurements = self.measurements[index].copy()
            for key, value in measurements.items():
                try:
                    value = torch.from_numpy(np.asarray([value, ]))
                    measurements[key] = value.float()
                except:
                    pass

            for sensor_name in self.sensor_data_names.keys():
                img_path = self.sensor_data_names[sensor_name][index]
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                if sensor_name == 'rgb':
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # [B, G, R] => [R, G, B]
                    # Apply the image transformation
                    if self.transform is not None:
                        boost = 1
                        img = self.transform(self.batch_read_number * boost, img)
                    else:
                        img = img.transpose(2, 0, 1)

                    img = img.astype(np.float)
                    img = torch.from_numpy(img).type(torch.FloatTensor)
                    img = img / 255.

                elif sensor_name == 'labels':
                    if self.transform is not None:
                        boost = 1
                        img = self.transform(self.batch_read_number * boost, img)
                    else:
                        img = img.transpose(2, 0, 1)

                    img = img[2, :, :]
                    if g_conf.LABELS_CLASSES != 13:
                        img = join_classes(img, g_conf.JOIN_CLASSES)

                    img = img.astype(np.float)
                    img = torch.from_numpy(img).type(torch.FloatTensor)
                    img = img / (g_conf.LABELS_CLASSES - 1)

                measurements[sensor_name] = img

            self.batch_read_number += 1

        except AttributeError:
            traceback.print_exc()
            print("Blank image")

            measurements = self.measurements[0].copy()
            for k, v in measurements.items():
                v = torch.from_numpy(np.asarray([v, ]))
                measurements[k] = v.float()
            measurements['steer'] = 0.0
            measurements['throttle'] = 0.0
            measurements['brake'] = 0.0
            measurements['rgb'] = np.zeros(3, 88, 200)  # TODO: This is hardcoded, should be able to read the image size

        return measurements

    def is_measurement_part_of_experiment(self, measurement_data):

        # If the measurement data is not removable it's because it's part of this experiment dataa
        return not self._check_remove_function(measurement_data, self._remove_params)

    def _get_final_measurement(self, speed, measurement_data, angle, available_measurements_dict):
        """
        Function to load the measurement with a certain angle and augmented direction.
        Also, it will choose if the brake is going to be present or if acceleration -1,1 is the default.

        Returns
            The final measurement dict
        """
        if angle != 0:
            measurement_augmented = self.augment_measurement(copy.copy(measurement_data),
                                                             angle,
                                                             3.6 * speed,
                                                             steer_name=available_measurements_dict['steer'])
        else:
            # We have to copy since it reference a file.
            measurement_augmented = copy.copy(measurement_data)
        # print(measurement_augmented)
        # sys.exit(1)
        # if 'gameTimestamp' in measurement_augmented:
        #     time_stamp = measurement_augmented['gameTimestamp']
        # else:
        #     time_stamp = measurement_augmented['elapsed_seconds']

        # final_measurement = {}
        # We go for every available measurement, previously tested
        # and update for the measurements vec that is used on the training.
        # for measurement, name_in_dataset in available_measurements_dict.items():
            # This is mapping the name of measurement in the target dataset
            # final_measurement.update({measurement: measurement_augmented[name_in_dataset]})

        # Add now the measurements that actually need some kind of processing
        # final_measurement.update({'speed_module': speed / g_conf.SPEED_FACTOR})
        # final_measurement.update({'directions': directions})
        # final_measurement.update({'game_time': time_stamp})
        measurement_augmented['forward_speed'] = measurement_data['forward_speed'] / g_conf.SPEED_FACTOR
        return measurement_augmented

    def _add_data_point(self, float_dicts, data_point, camera_angle):
        """
        Add a data point to the vector that is the full dataset
        Args:
            float_dicts:
            data_point:
            camera_angle:
        Returns:
        """
        # Augment the steering if the camera angle is not 0
        new_data_copy = copy.deepcopy(data_point)
        if camera_angle != 0:
            new_data_copy['measurements']['steer'] = self.augment_steering(camera_angle,
                                                                           new_data_copy['measurements']['steer'],
                                                                           new_data_copy['measurements']['forward_speed']* 3.6)  # TODO: speed
            new_data_copy['measurements']['forward_speed'] = data_point['measurements']['forward_speed'] / g_conf.SPEED_FACTOR
            float_dicts.append(new_data_copy['measurements'])

    def _pre_load_image_folders(self, path):
        """
        Pre load the image folders for each episode, keep in mind that we only take
        the measurements that we think that are interesting for now.

        Args
            the path for the dataset

        Returns
            sensor data names: it is a vector with n dimensions being one for each sensor modality
            for instance, rgb only dataset will have a single vector with all the image names.
            float_data: all the wanted float data is loaded inside a vector, that is a vector
            of dictionaries.

        """
        # sensor_data_names = {}
        # for sensor in g_conf.SENSORS.keys():
        #     sensor_data_names[sensor.split('_')[0]] = []
        #
        # if path == 'validation':
        #     jsonfile = g_conf.EXPERIENCE_FILE_VALID  # TODO: set this default and check options for path
        # else:
        #     jsonfile = g_conf.EXPERIENCE_FILE
        #
        # # We check one image at least to see if it matches the size expected by the network
        # checked_image = True
        # float_dicts = []
        # env_batch = CEXP(jsonfile, params=None, execute_all=True, ignore_previous_execution=True)
        # # We start the server without docker
        # env_batch.start(no_server=True, agent_name='Agent')
        #
        # # We count the environments that are read
        # for env in env_batch:
        #     print(f'Container name: {env}')
        #     try:
        #         env_data = env.get_data()
        #     except NoDataGenerated:
        #         print('\tNo data generated for this environment.')
        #         sys.exit(1)
        #
        #     for exp in env_data:
        #         print(f'\tClient: {exp[1]}')
        #         for batch in exp[0]:
        #             print(f'\t\tBatch: {batch[1]} of length {len(batch[0])}')
        #             for data_point in batch[0]:
        #                 # Delete some non-floatable cases
        #                 try:
        #                     del data_point['measurements']['ego_actor']
        #                     del data_point['measurements']['opponents']
        #                     del data_point['measurements']['lane']
        #                     del data_point['measurements']['hand_brake']
        #                     del data_point['measurements']['reverse']  # TODO: check compatibility with current data
        #                 except NameError:
        #                     pass
        #
        #                 self._add_data_point(float_dicts, data_point, 0.0)
        #                 if g_conf.AUGMENT_LATERAL_STEERINGS > 0.0:
        #                     self._add_data_point(float_dicts, data_point, -30.0)
        #                     self._add_data_point(float_dicts, data_point, 30.0)
        #
        #                 for sensor in g_conf.SENSORS.keys():
        #                     if not checked_image:
        #                         if not check_size(data_point[sensor], g_conf.SENSORS[sensor]):
        #                             raise RuntimeError('Unexpected image size for the network!')
        #                         checked_image = True
        #
        #                     sensor_data_names[sensor.split('_')[0]].append(data_point[sensor])
        #                     if g_conf.AUGMENT_LATERAL_STEERINGS > 0.0:
        #                         sensor_data_names[sensor.split('_')[0]].append(
        #                             data_point[sensor.split('_')[0]+'_left'])
        #                         sensor_data_names[sensor.split('_')[0]].append(
        #                             data_point[sensor.split('_')[0] + '_right'])


        # ==================================
        containers_list = glob.glob(os.path.join(path, '*route*'))  # TODO: now episode_* is Container_*
        sort_nicely(containers_list)
        # Do a check if the episodes list is empty
        if len(containers_list) == 0:
            raise ValueError(f"There are no episodes on the training dataset folder: {path}")

        # We will check one image to see if it matches the size expected by the network
        checked_image = False
        sensor_data_names = {}
        float_dicts = []

        number_of_hours_pre_loaded = 0

        # Now we do a check to try to find all the
        for container in containers_list:
            print(f'Container name: {container}')
            available_measurements_dict = data_parser.check_available_measurements(container)
            if number_of_hours_pre_loaded > g_conf.NUMBER_OF_HOURS:
                # The number of wanted hours achieved
                break
            client_list = glob.glob(os.path.join(container, '**/0'))  # TODO: Client_*
            for client in client_list:
                # Get all the measurements from this client
                measurements_list = glob.glob(os.path.join(client, 'measurements*'), recursive=True)  # TODO: can_bus*
                sort_nicely(measurements_list)

                if len(measurements_list) == 0:
                    print("Empty client")
                    continue

                # A simple count to keep track how many measurements were added this episode.
                count_added_measurements = 0

                for measurement in tqdm(measurements_list):
                    data_point_number = measurement.split('_')[-1].split('.')[0]  # /path/to/measurements_0019.json => 0019
                    with open(measurement) as f:
                        measurement_data = json.load(f)
                    # Delete some non-floatable cases
                    try:
                        del measurement_data['ego_actor']
                        del measurement_data['opponents']
                        del measurement_data['lane']
                        del measurement_data['hand_brake']
                        del measurement_data['reverse']  # TODO: check compatibility with current data
                    except NameError:
                        pass
                    # depending on the configuration file, we eliminated the kind of measurements
                    # that are not going to be used for this experiment
                    # We extract the interesting subset from the measurement dict
                    speed = data_parser.get_speed(measurement_data)

                    for sensor in g_conf.SENSORS.keys():

                        final_measurement = self._get_final_measurement(speed, measurement_data, 0,
                                                                        available_measurements_dict)

                        sensor_name = sensor.split('_')[0]
                        if self.is_measurement_part_of_experiment(final_measurement):
                            float_dicts.append(final_measurement)
                            rgb = glob.glob(os.path.join(container, f'**/rgb_central{data_point_number}.png'),
                                            recursive=True)[0]  # TODO: we have multiple clients with same image name
                            if not checked_image:
                                if not check_size(rgb, g_conf.SENSORS[sensor]):
                                    raise RuntimeError('Unexpected image size for the network!')
                                checked_image = True

                            if sensor_name in sensor_data_names:
                                sensor_data_names[sensor_name].append(rgb)
                            else:
                                sensor_data_names[sensor_name] = [rgb]
                            count_added_measurements += 1

                        # We do measurements for the left side camera
                        # We convert the speed to KM/h for the augmentation
                        # We extract the interesting subset from the measurement dict
                        final_measurement = self._get_final_measurement(speed, measurement_data, -30.0,
                                                                        available_measurements_dict)
                        if self.is_measurement_part_of_experiment(final_measurement):
                            float_dicts.append(final_measurement)
                            rgb = glob.glob(os.path.join(container, f'**/rgb_left{data_point_number}.png'),
                                            recursive=True)[0]  # TODO: we have multiple clients with same image name

                            if sensor_name in sensor_data_names:
                                sensor_data_names[sensor_name].append(rgb)
                            else:
                                sensor_data_names[sensor_name] = [rgb]
                            count_added_measurements += 1

                        # We do measurements augmentation for the right side cameras

                        final_measurement = self._get_final_measurement(speed, measurement_data, 30.0,
                                                                        available_measurements_dict)

                        if self.is_measurement_part_of_experiment(final_measurement):
                            float_dicts.append(final_measurement)
                            rgb = glob.glob(os.path.join(container, f'**/rgb_right{data_point_number}.png'),
                                            recursive=True)[0]  # TODO: we have multiple clients with same image name

                            if sensor_name in sensor_data_names:
                                sensor_data_names[sensor_name].append(rgb)
                            else:
                                sensor_data_names[sensor_name] = [rgb]
                            count_added_measurements += 1

            # Check how many hours were actually added
            number_of_hours_pre_loaded += (float(count_added_measurements / 20.0) / 3600.0)  # TODO: add FPS to config?
            print(f"Loaded {number_of_hours_pre_loaded} hours of data")
        # ==================================

        # Make the path to save the pre loaded datasets
        if not os.path.exists('_preloads'):
            os.mkdir('_preloads')
        # If there is a name we saved the preloaded data
        if self.preload_name is not None:
            np.save(os.path.join('_preloads', self.preload_name), [sensor_data_names, float_dicts])

        return sensor_data_names, float_dicts

    @staticmethod
    def augment_directions(directions):

        if directions == 2.0:
            if random.randint(0, 100) < 20:
                directions = random.choice([3.0, 4.0, 5.0])

        return directions

    @staticmethod
    def augment_steering(camera_angle, steer, speed):
        """
            Apply the steering physical equation to augment for the lateral cameras steering
        Args:
            camera_angle: the angle of the camera
            steer: the central steering
            speed: the speed that the car is going

        Returns:
            the augmented steering

        """
        time_use = 1.0
        car_length = 6.0

        pos = camera_angle > 0.0
        neg = camera_angle <= 0.0
        # You should use the absolute value of speed
        speed = math.fabs(speed)
        rad_camera_angle = math.radians(math.fabs(camera_angle))
        val = g_conf.AUGMENT_LATERAL_STEERINGS * (
            math.atan((rad_camera_angle * car_length) / (time_use * speed + 0.05))) / 3.1415
        steer -= pos * min(val, 0.3)
        steer += neg * min(val, 0.3)

        steer = min(1.0, max(-1.0, steer))

        # print('Angle', camera_angle, ' Steer ', old_steer, ' speed ', speed, 'new steer', steer)
        return steer

    def augment_measurement(self, measurements, angle, speed, steer_name='steer'):
        """
            Augment the steering of a measurement dict

        """
        new_steer = self.augment_steering(angle, measurements[steer_name],
                                          speed)
        measurements[steer_name] = new_steer
        return measurements

    def controls_position(self):
        return np.where(self.meta_data[:, 0] == b'control')[0][0]

    """
        Methods to interact with the dataset attributes that are used for training.
    """

    @staticmethod
    def extract_targets(data):
        """
        Method used to get to know which positions from the dataset are the targets
        for this experiments
        Args:
            data: the set of all float data got from the dataset

        Returns:
            the float data that is actually targets

        Raises
            value error when the configuration set targets that didn't exist in metadata
        """
        targets_vec = []
        for target_name in g_conf.TARGETS:
            targets_vec.append(data[target_name])

        return torch.cat(targets_vec, 1)

    @staticmethod
    def extract_inputs(data):
        """
        Method used to get to know which positions from the dataset are the inputs
        for this experiments
        Args:
            data: the set of all float data got from the dataset

        Returns:
            the float data that is actually targets

        Raises
            value error when the configuration set targets that didn't exist in metadata
        """
        inputs_vec = []
        for input_name in g_conf.INPUTS:
            if len(data[input_name].size()) > 2:
                inputs_vec.append(torch.squeeze(data[input_name]))
            else:
                inputs_vec.append(data[input_name])

        return torch.cat(inputs_vec, 1)

    @staticmethod
    def extract_intentions(data):
        """
        Method used to get to know which positions from the dataset are the inputs
        for this experiments
        Args:
            data: the set of all float data got from the dataset

        Returns:
            the float data that is actually targets

        Raises
            value error when the configuration set targets that didn't exist in metadata
        """
        inputs_vec = []
        for input_name in g_conf.INTENTIONS:
            inputs_vec.append(data[input_name])

        return torch.cat(inputs_vec, 1)

