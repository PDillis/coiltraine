import os
import glob
import traceback
import collections

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

from cexp.cexp import CEXP
from cexp.env.scenario_identification import identify_scenario
from cexp.env.environment import NoDataGenerated

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


def convert_scenario_name_number(measurements):
    scenario = identify_scenario(measurements['distance_intersection'],
                                 measurements['road_angle'])
    if scenario == 'S0_lane_following':
        return [1, 0, 0, 0]
    elif scenario == 'S1_lane_following_curve':
        return [0, 1, 0, 0]
    elif scenario == 'S2_before_intersection':
        return [0, 0, 1, 0]
    elif scenario == 'S3_intersection':
        return [0, 0, 0, 1]
    else:
        raise ValueError(f'Unexpected scenario identified: {scenario}')


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
    # print(" WEATHER OF EPISODE ", metadata['weather'])
    return int(metadata['weather'])


class CoILDataset(Dataset):
    """ The conditional imitation learning dataset"""

    def __init__(self, root_dir, transform=None, preload_name=None, process_type=None):
        # Setting the root directory for this dataset
        self.root_dir = root_dir
        # We add to the preload name all the remove labels
        if eval(str(g_conf.REMOVE)) is not None:
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
                print(f'\t======> Total {key} images: {len(self.sensor_data_names[key])}')
            print(f'\t======> Total measurements: {len(self.measurements)}')
        else:
            print('Creating NPY preload...')
            self.sensor_data_names, self.measurements = self._pre_load_image_folders(root_dir, process_type)
            for key in self.sensor_data_names.keys():
                print(f'\t======> Total {key} images: {len(self.sensor_data_names[key])}')
            print(f'\t======> Total measurements: {len(self.measurements)}')

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
            print("Blank image!")
            measurements = self.measurements[0].copy()
            for k, v in measurements.items():
                v = torch.from_numpy(np.asarray([v, ]))
                measurements[k] = v.float()
            measurements['steer'] = 0.0
            measurements['throttle'] = 0.0
            measurements['brake'] = 0.0
            measurements['rgb'] = np.zeros(g_conf.SENSORS['rgb'])

        return measurements

    def is_measurement_part_of_experiment(self, measurement_data):

        # If the measurement data is not removable it's because it's part of this experiment dataa
        return not self._check_remove_function(measurement_data, self._remove_params)

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

    def _add_data_point(self, float_dicts, data_point, camera_angle):
        """
        Add a data point to the vector that is the full dataset
        Args:
            float_dicts:
            data_point:
            camera_angle: the augmentation angle to be applied to the steering
        Returns:
        """
        new_data_copy = copy.deepcopy(data_point)
        # If angle is not 0, use the augment_steering function
        if camera_angle != 0:
            # Convert the speed from m/s to km/h ???
            new_data_copy['measurements']['steer'] = self.augment_steering(camera_angle,
                                                                           new_data_copy['measurements']['steer'],
                                                                           new_data_copy['measurements']['speed'] * 3.6)

        new_data_copy['measurements']['speed'] = data_point['measurements']['speed'] / g_conf.SPEED_FACTOR  # normalize
        float_dicts.append(new_data_copy['measurements'])

    def _pre_load_image_folders(self, path, process_type):
        """
        The old function (below) was too slow to preload the dataset. Hence, we will use C-EXP since we will be able to
        load the data path and image information much faster by using the GPU.
        Args:
            path: dataset path
            process_type: 'training' or 'validation'
        Returns:
            sensor_data_names: vector containing each sensor modality ('rgb', 'lidar', etc.) containing the paths to the
            sensors or cameras in the case for rgb images
            float_dicts: dictionary containing the float data for each of these sensors
        """
        # Empty dicts and vector to append the data we will return
        sensor_data_names = {}
        float_dicts = []

        for sensor in g_conf.SENSORS.keys():
            # We will have 'rgb_central', 'rgb_right', etc., so we just care for the type of sensor, i.e., 'rgb'
            sensor_data_names[sensor.split('_')[0]] = []

        if process_type == 'validation':
            experience_json = g_conf.EXPERIENCE_FILE_VALID
        elif process_type == 'train':
            experience_json = g_conf.EXPERIENCE_FILE
        else:
            raise Exception("Invalid name for process_type, chose from (train, validation)")

        # We will check one image to see if it is what the network expects
        checked_image = False

        container_batch = CEXP(experience_json, params=None, execute_all=True, ignore_previous_execution=True)
        # Start the server without Docker
        container_batch.start(no_server=True, agent_name='Client')

        for container in container_batch:
            print(f'Container name: {container}')
            try:
                container_data = container.get_data()  # Returns a way to read all the data properly
            except NoDataGenerated:
                print("No data generated for this container!")
            else:
                for client in container_data:  # Client_354, Client_355, ....
                    for data in client[0]:
                        for data_point in data[0]:
                            # We delete non-float cases; will depend on the data that has been saved (see can_bus.json)
                            del data_point['measurements']['hand_brake']
                            del data_point['measurements']['reverse']
                            del data_point['measurements']['ego_position']
                            del data_point['measurements']['route_nodes_xyz']
                            del data_point['measurements']['route_nodes']

                            self._add_data_point(float_dicts, data_point, 0)  # Central camera
                            if g_conf.AUGMENT_LATERAL_STEERINGS > 0:
                                self._add_data_point(float_dicts, data_point, -30)  # left camera
                                self._add_data_point(float_dicts, data_point, 30)

                            for sensor in g_conf.SENSORS.keys():
                                # check one image
                                if not checked_image:
                                    if not check_size(data_point[f'{sensor}_central'], g_conf.SENSORS[sensor]):
                                        raise RuntimeError('Image size mismatch for configuration and training data!')
                                    checked_image = True

                                # Add paths for central, left, and right sensors (if there are augmentations)
                                sensor_data_names[sensor].append(data_point[f'{sensor}_central'])
                                if g_conf.AUGMENT_LATERAL_STEERINGS > 0:
                                    sensor_data_names[sensor].append(data_point[f'{sensor}_left'])
                                    sensor_data_names[sensor].append(data_point[f'{sensor}_right'])

        # Create path for the preloaded NPY files
        if not os.path.exists('_preloads'):
            os.mkdir('_preloads')
        # If there is a name, we save the preloaded data
        if self.preload_name is not None:
            np.save(os.path.join('_preloads', self.preload_name), [sensor_data_names, float_dicts])

        return sensor_data_names, float_dicts

    ##### Legacy code #####
    def augment_measurement(self, measurements, angle, speed, steer_name='steer'):
        """
            Augment the steering of a measurement dict

        """
        new_steer = self.augment_steering(angle, measurements[steer_name],
                                          speed)
        measurements[steer_name] = new_steer
        return measurements

    def _get_final_measurement(self, speed, measurement_data, angle):
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
                                                             'steer')
        else:
            # We have to copy since it reference a file.
            measurement_augmented = copy.copy(measurement_data)

        # Add now the measurements that actually need some kind of processing
        # final_measurement.update({'speed_module': speed / g_conf.SPEED_FACTOR})
        # final_measurement.update({'directions': directions})
        # final_measurement.update({'game_time': time_stamp})
        measurement_augmented['speed'] = measurement_data['speed'] / g_conf.SPEED_FACTOR
        return measurement_augmented

    def _pre_load_image_folders_old(self, path):
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
        containers_list = glob.glob(os.path.join(path, 'Container_*'))
        sort_nicely(containers_list)
        # Do a check if the episodes list is empty
        if len(containers_list) == 0:
            raise ValueError(f"There are no containers on the training dataset folder: {path}")

        # We will check one image to see if it matches the size expected by the network
        checked_image = False
        sensor_data_names = {}
        float_dicts = []

        number_of_hours_pre_loaded = 0

        # Now we do a check to try to find all the
        for container in containers_list:
            print(f'Container name: {container}')
            if number_of_hours_pre_loaded > g_conf.NUMBER_OF_HOURS:
                # The number of wanted hours achieved
                break
            # A simple count to keep track how many measurements were added this episode.
            count_added_measurements = 0
            # We may have more than one client for each container, so the data_point_number might clash later
            client_list = glob.glob(os.path.join(container, '**/Client_*'), recursive=True)

            for client in client_list:
                # Get all the measurements from this client
                measurements_list = glob.glob(os.path.join(client, 'can_bus*'))
                sort_nicely(measurements_list)

                if len(measurements_list) == 0:
                    print("Empty client")
                    continue

                for measurement in tqdm(measurements_list):
                    data_point_number = os.path.splitext(measurement)[0][-6:]  # /pth/to/can_bus000019.json => 000019
                    with open(measurement) as f:
                        measurement_data = json.load(f)
                    # Delete some non-floatable cases
                    # depending on the configuration file, we eliminated the kind of measurements
                    # that are not going to be used for this experiment
                    del measurement_data['hand_brake']
                    del measurement_data['reverse']  # TODO: not relevant now, but we might be interested later on
                    del measurement_data['ego_position']
                    del measurement_data['route_nodes_xyz']
                    del measurement_data['route_nodes']

                    # We extract the interesting subset from the measurement dict
                    speed = data_parser.get_speed(measurement_data)

                    for sensor in g_conf.SENSORS.keys():
                        # We will go through each of the cameras
                        cameras = (('central', 0), ('left', -30.0), ('right', 30.0))
                        sensor_name = sensor.split('_')[0]
                        for cam in cameras:
                            # We do measurements for the three cameras
                            # We convert the speed to KM/h for the augmentation
                            # We extract the interesting subset from the measurement dict
                            final_measurement = self._get_final_measurement(speed, measurement_data, cam[1])
                            if self.is_measurement_part_of_experiment(final_measurement):
                                float_dicts.append(final_measurement)
                                sensor_path = glob.glob(
                                    os.path.join(client, f'**/{sensor_name}_{cam[0]}{data_point_number}.png'),
                                    recursive=True)
                                if len(sensor_path) == 0:
                                    continue
                                if not checked_image:
                                    if not check_size(*sensor_path, g_conf.SENSORS[sensor]):
                                        raise RuntimeError('Unexpected image size for the network!')
                                    checked_image = True

                                if sensor_name in sensor_data_names:
                                    sensor_data_names[sensor_name].append(*sensor_path)
                                else:
                                    sensor_data_names[sensor_name] = [*sensor_path]
                                count_added_measurements += 1

            # Check how many hours were actually added
            number_of_hours_pre_loaded += (float(count_added_measurements / g_conf.TRAIN_DATA_FPS) / 3600.0)
            print(f"Loaded {number_of_hours_pre_loaded} hours of data")

        # Make the path to save the pre loaded datasets
        if not os.path.exists('_preloads'):
            os.mkdir('_preloads')
        # If there is a name we saved the preloaded data
        if self.preload_name is not None:
            np.save(os.path.join('_preloads', self.preload_name), [sensor_data_names, float_dicts])

        return sensor_data_names, float_dicts

    #######################

    @staticmethod
    def augment_directions(directions):

        if directions == 2.0:
            if random.randint(0, 100) < 20:
                directions = random.choice([3.0, 4.0, 5.0])

        return directions

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
