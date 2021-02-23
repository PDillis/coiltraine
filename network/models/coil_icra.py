from logger import coil_logger
import torch.nn as nn
import torch
import importlib

from configs import g_conf
from coilutils.general import command_number_to_index

from .building_blocks import Conv
from .building_blocks import Branching
from .building_blocks import FC
from .building_blocks import Join


class CoILICRA(nn.Module):

    def __init__(self, params):

        super(CoILICRA, self).__init__()
        self.params = params

        number_first_layer_channels = 0

        # If we fuse more than one frame, then the first layer will be a concatenation of
        # the channels of this first layer [3, w, h] (e.g., 2 RGB images->3+3=6)
        for _, sizes in g_conf.SENSORS.items():
            number_first_layer_channels += sizes[0] * g_conf.NUMBER_FRAMES_FUSION

        # Get one item from the dict
        sensor_input_shape = next(iter(g_conf.SENSORS.values()))  # [3, 300, 400]
        sensor_input_shape = [number_first_layer_channels, sensor_input_shape[1],
                              sensor_input_shape[2]]  # replace the above result on the channels here

        # For this case we check if the perception layer is of the type "conv"
        if 'conv' in params['perception']:
            perception_convs = Conv(params={
                'channels': [number_first_layer_channels] + params['perception']['conv']['channels'],
                'kernels': params['perception']['conv']['kernels'],
                'strides': params['perception']['conv']['strides'],
                'dropouts': params['perception']['conv']['dropouts'],
                'end_layer': True})

            perception_fc = FC(params={
                'neurons': [perception_convs.get_conv_output(sensor_input_shape)] + params['perception']['fc']['neurons'],
                'dropouts': params['perception']['fc']['dropouts'],
                'end_layer': False})

            self.perception = nn.Sequential(*[perception_convs, perception_fc])

            number_output_neurons = params['perception']['fc']['neurons'][-1]

        elif 'res' in params['perception']:  # pre defined residual networks
            resnet_module = importlib.import_module('network.models.building_blocks.resnet')
            resnet_module = getattr(resnet_module, params['perception']['res']['name'])
            self.perception = resnet_module(pretrained=g_conf.PRE_TRAINED,
                                            num_classes=params['perception']['res']['num_classes'])

            number_output_neurons = params['perception']['res']['num_classes']

        else:

            raise ValueError("Invalid perception layer type; only convolutional ('conv') or ResNet-based ('res') "
                             "are allowed)")

        self.intermediate_layers = None

        self.measurements = FC(params={'neurons': [len(g_conf.INPUTS)] + params['measurements']['fc']['neurons'],
                                       'dropouts': params['measurements']['fc']['dropouts'],
                                       'end_layer': False})

        self.join = Join(
            params={
                'after_process': FC(params={
                    'neurons': [params['measurements']['fc']['neurons'][-1] + number_output_neurons] +
                               params['join']['fc']['neurons'],
                    'dropouts': params['join']['fc']['dropouts'],
                    'end_layer': False}),
                'mode': 'cat'})

        self.speed_branch = FC(params={
            'neurons': [params['join']['fc']['neurons'][-1]] + params['speed_branch']['fc']['neurons'] + [1],
            'dropouts': params['speed_branch']['fc']['dropouts'] + [0.0],
            'end_layer': True})

        # Create the fc vector separately
        branch_fc_vector = []
        for i in range(params['branches']['number_of_branches']):
            branch_fc_vector.append(FC(params={'neurons': [params['join']['fc']['neurons'][-1]] +
                                                          params['branches']['fc']['neurons'] + [len(g_conf.TARGETS)],
                                               'dropouts': params['branches']['fc']['dropouts'] + [0.0],
                                               'end_layer': True}))

        self.branches = Branching(branch_fc_vector)  # Here we set branching automatically

        # Weight initialization for the convolutional perception modules
        if 'conv' in params['perception']:
            for m in self.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0.1)
        # Init for the rest of the network
        else:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0.1)

    def forward(self, x, a):
        """ ###### APPLY THE PERCEPTION MODULE """
        x, self.intermediate_layers = self.perception(x)
        # Not a variable, just to store intermediate layers for future visualization
        # self.intermediate_layers = inter

        """ APPLY THE MEASUREMENT MODULE """
        m = self.measurements(a)
        """ Join measurements and perception"""
        j = self.join(x, m)

        branch_outputs = self.branches(j)

        speed_branch_output = self.speed_branch(x)

        # We concatenate speed with the rest.
        return branch_outputs + [speed_branch_output]

    def forward_branch(self, x, a, branch_number):
        """
        Do a forward operation and return a single branch.

        Args:
            x: the image input (torch.squeeze(data['rgb']))
            a: speed measurement (dataset.extract_inputs(data))
            branch_number: the branch number to be returned (data['directions'])

        Returns:
            the forward operation on the selected branch

        """
        output_vec = torch.stack(self.forward(x, a)[0:self.params['branches']['number_of_branches']])

        return self.extract_branch(output_vec, branch_number)

    def get_perception_layers(self, x):
        return self.perception.get_layers_features(x)

    @staticmethod
    def extract_branch(output_vec, branch_number):
        # Extract
        branch_number = command_number_to_index(branch_number)

        if len(branch_number) > 1:
            branch_number = torch.squeeze(branch_number.type(torch.cuda.LongTensor))
        else:
            branch_number = branch_number.type(torch.cuda.LongTensor)

        branch_number = torch.stack([branch_number,
                                     torch.cuda.LongTensor(range(0, len(branch_number)))])

        return output_vec[branch_number[0], branch_number[1], :]
