import glob
import os
import json
import numpy as np
"""
Module used to check attributes existent on data before incorporating them
to the coil dataset
"""


def orientation_vector(measurement_data):
    pitch = np.deg2rad(measurement_data['ego_actor']['orientation'][0])
    yaw = np.deg2rad(measurement_data['ego_actor']['orientation'][2])
    orientation = np.array([np.cos(pitch)*np.cos(yaw), np.cos(pitch)*np.sin(yaw), np.sin(pitch)])
    return orientation


def forward_speed(measurement_data):
    vel_np = np.array([measurement_data['ego_actor']['velocity'][0],
                       measurement_data['ego_actor']['velocity'][1],
                       measurement_data['ego_actor']['velocity'][2]])
    speed = np.dot(vel_np, orientation_vector(measurement_data))

    return speed


def get_speed(measurement_data):
    """
    Extract the proper speed from the measurement data dict. This will come from a {measurements/can_bus_%6d.json.
    """
    if 'forward_speed' in measurement_data:
        return measurement_data['forward_speed']
    elif 'speed' in measurement_data:
        return measurement_data['speed']
    elif 'ego_actor' in measurement_data:  # We have a 0.9.X data here; TODO: we should save this in TM data
        return forward_speed(measurement_data)
    else:  # If the forward speed is not on the dataset it is because speed is zero.
        return 0.0


def check_available_measurements(client):
    """ Try to automatically check the measurements
        The ones named 'steer' are probably the steer for the vehicle
        This needs to be made more general to avoid possible mistakes on dataset reading
    """
    measurements_list = glob.glob(os.path.join(client, '**/can_bus*'), recursive=True)
    assert len(measurements_list) > 0, 'No measurements in the episode!'
    # Open a sample measurement
    with open(measurements_list[0]) as f:
        measurement_data = json.load(f)

    available_measurements = {}
    for meas_name in measurement_data.keys():

        # Add steer
        if 'steer' in meas_name and 'noise' not in meas_name:
            available_measurements.update({'steer': meas_name})

        # Add Throttle
        if 'throttle' in meas_name and 'noise' not in meas_name:
            available_measurements.update({'throttle': meas_name})

        # Add brake ( Not hand brake)
        if 'brake' in meas_name and 'noise' not in meas_name and 'hand' not in meas_name:
            available_measurements.update({'brake': meas_name})

        if 'speed' in meas_name and 'noise' not in meas_name:
            available_measurements.update({'speed': meas_name})

    return available_measurements
