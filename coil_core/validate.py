import os
import time
import sys
import random

import torch
import traceback
import dlib

# What do we define as a parameter what not.

from configs import g_conf, set_type_of_process, merge_with_yaml
from network import CoILModel, Loss
from input import CoILDataset, Augmenter
from logger import coil_logger
from coilutils.checkpoint_schedule import get_latest_evaluated_checkpoint, is_next_checkpoint_ready,\
    maximum_checkpoint_reached, get_next_checkpoint
from coilutils.general import save_output


def write_waypoints_output(iteration, output):

    for i in range(g_conf.BATCH_SIZE):
        steer = 0.7 * output[i][3]

        if steer > 0:
            steer = min(steer, 1)
        else:
            steer = max(steer, -1)

        coil_logger.write_on_csv(iteration, [steer, output[i][1], output[i][2]])


def write_regular_output(iteration, output):
    for i in range(len(output)):
        coil_logger.write_on_csv(iteration, [output[i][0], output[i][1], output[i][2]])


# The main function maybe we could call it with a default name
def execute(gpu, exp_batch, exp_alias, suppress_output):
    latest = None
    try:
        # We set the visible cuda devices
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu

        # At this point the log file with the correct naming is created.
        merge_with_yaml(os.path.join('configs', exp_batch, f'{exp_alias}.yaml'))
        # The validation dataset is always fully loaded, so we fix a very high number of hours
        g_conf.NUMBER_OF_HOURS = 10000
        set_type_of_process(process_type='validation', param=g_conf.VAL_DATASET_NAME)

        # Save the output to a file if so desired
        if suppress_output:
            save_output(exp_alias)

        # Define the dataset. This structure has the __get_item__ redefined in a way
        # that you can access the HDFILES positions from the root directory as a in a vector.
        full_dataset = os.path.join(os.environ["COIL_DATASET_PATH"], g_conf.VAL_DATASET_NAME)
        augmenter = Augmenter(None)
        # Definition of the dataset to be used. Preload name is just the validation data name
        dataset = CoILDataset(full_dataset,
                              transform=augmenter,
                              preload_name=g_conf.VAL_DATASET_NAME,
                              process_type='validation')

        # Creates the sampler, this part is responsible for managing the keys. It divides
        # all keys depending on the measurements and produces a set of keys for each bach.

        # The data loader is the multi threaded module from pytorch that release a number of
        # workers to get all the data.
        data_loader = torch.utils.data.DataLoader(dataset,
                                                  batch_size=g_conf.BATCH_SIZE,
                                                  shuffle=False,
                                                  num_workers=g_conf.NUMBER_OF_LOADING_WORKERS,
                                                  pin_memory=True)

        model = CoILModel(g_conf.MODEL_TYPE, g_conf.MODEL_CONFIGURATION).cuda()
        # The window used to keep track of the trainings
        l1_window = []
        latest = get_latest_evaluated_checkpoint()
        if latest is not None:  # When latest is noe
            l1_window = coil_logger.recover_loss_window(g_conf.VAL_DATASET_NAME, None)

        best_mse = 1000
        best_error = 1000
        best_loss = 1000
        best_mse_iter = 0
        best_error_iter = 0
        best_loss_iter = 0

        print(20 * '#')
        print('Starting validation!')
        print(20 * '#')

        # Check if the maximum checkpoint for validating has been reached
        while not maximum_checkpoint_reached(latest):
            # Wait until the next checkpoint is ready (assuming this is run whilst training the model)
            if is_next_checkpoint_ready(g_conf.TEST_SCHEDULE):
                # Get next checkpoint for validation according to the test schedule and load it
                latest = get_next_checkpoint(g_conf.TEST_SCHEDULE)
                checkpoint = torch.load(os.path.join('_logs', exp_batch, exp_alias, 'checkpoints', f'{latest}.pth'))
                checkpoint_iteration = checkpoint['iteration']

                model.load_state_dict(checkpoint['state_dict'])
                model.eval()  # Turn off dropout and batchnorm (if any)
                print(f"Validation loaded, checkpoint {checkpoint_iteration}")

                # Main metric will be the used loss for training the network
                criterion = Loss(g_conf.LOSS_FUNCTION)
                checkpoint_average_loss = 0
                # Auxiliary metrics
                errors = []
                checkpoint_average_error = 0
                checkpoint_average_mse = 0
                # Counter
                iteration_on_checkpoint = 0

                with torch.no_grad():  # save some computation/memory
                    for data in data_loader:
                        # Compute the forward pass on a batch from the validation dataset
                        controls = data['directions'].cuda()
                        img = torch.squeeze(data['rgb']).cuda()
                        speed = dataset.extract_inputs(data).cuda()  # this might not always be speed

                        # For auxiliary metrics
                        output = model.forward_branch(img, speed, controls)

                        # For the loss function
                        branches = model(img, speed)
                        loss_function_params = {
                            'branches': branches,
                            'targets': dataset.extract_targets(data).cuda(),
                            'controls': controls,
                            'inputs': speed,
                            'branch_weights': g_conf.BRANCH_LOSS_WEIGHT,
                            'variable_weights': g_conf.VARIABLE_WEIGHT
                        }
                        # It could be either waypoints or direct control
                        if 'waypoint1_angle' in g_conf.TARGETS:
                            write_waypoints_output(checkpoint_iteration, output)
                        else:
                            write_regular_output(checkpoint_iteration, output)

                        loss, _ = criterion(loss_function_params)
                        loss = loss.data.tolist()

                        err = output - dataset.extract_targets(data).cuda()
                        errors.append(err.cpu().data.tolist())
                        error = torch.abs(err)
                        mse = torch.mean(err**2).data.tolist()
                        mean_error = torch.mean(error).data.tolist()

                        # Log a random position
                        position = random.randint(0, len(output.data.tolist()) - 1)

                        coil_logger.add_message('Iterating', {
                            'Checkpoint': latest,
                            'Iteration': f'{iteration_on_checkpoint * g_conf.BATCH_SIZE}/{len(dataset)}',
                            f'Validation Loss ({g_conf.LOSS_FUNCTION})': loss,
                            'MeanError': mean_error,
                            'MSE': mse,
                            'Output': output[position].data.tolist(),
                            'GroundTruth': dataset.extract_targets(data)[position].data.tolist(),
                            'Error': error[position].data.tolist(),
                            'Inputs': dataset.extract_inputs(data)[position].data.tolist()
                        }, latest)

                        # We get the average with a growing list of values
                        # Thanks to John D. Cook: http://www.johndcook.com/blog/standard_deviation/
                        iteration_on_checkpoint += 1
                        checkpoint_average_loss += (loss - checkpoint_average_loss) / iteration_on_checkpoint
                        checkpoint_average_error += (mean_error - checkpoint_average_error) / iteration_on_checkpoint
                        checkpoint_average_mse += (mse - checkpoint_average_mse) / iteration_on_checkpoint
                        print(f"\rProgress: {100 * iteration_on_checkpoint * g_conf.BATCH_SIZE / len(dataset):3.4f}% - "
                              f"Average Loss ({g_conf.LOSS_FUNCTION}): {checkpoint_average_loss:.16f} - "
                              f"Average Error: {checkpoint_average_error:.16f} - "
                              f"Average MSE: {checkpoint_average_mse:.16f}", end='')

                """
                    ########
                    Finish a round of validation, write results, wait for the next
                    ########
                """
                coil_logger.add_scalar('Checkpoint MSE', checkpoint_average_mse, latest, True)
                coil_logger.add_scalar('Checkpoint Error', checkpoint_average_error, latest, True)
                coil_logger.add_scalar(f'Checkpoint Loss ({g_conf.LOSS_FUNCTION})',
                                       checkpoint_average_loss, latest, True)

                coil_logger.add_histogram('Error (pred - real)', errors, latest)

                if checkpoint_average_mse < best_mse:
                    best_mse = checkpoint_average_mse
                    best_mse_iter = latest

                if checkpoint_average_error < best_error:
                    best_error = checkpoint_average_error
                    best_error_iter = latest

                if checkpoint_average_loss < best_loss:
                    best_loss = checkpoint_average_error
                    best_loss_iter = latest

                coil_logger.add_message('Iterating', {'Summary': {'Loss': checkpoint_average_loss,
                                                                  'Error': checkpoint_average_error,
                                                                  'MSE': checkpoint_average_mse,
                                                                  'BestLoss': best_loss,
                                                                  'BestLossCheckpoint': best_loss_iter,
                                                                  'BestError': best_error,
                                                                  'BestErrorCheckpoint': best_error_iter,
                                                                  'BestMSE': best_mse,
                                                                  'BestMSECheckpoint': best_mse_iter},
                                                      'Checkpoint': latest},
                                        latest)

                l1_window.append(checkpoint_average_error)
                coil_logger.write_on_error_csv(g_conf.VAL_DATASET_NAME,
                                               [checkpoint_average_loss,
                                                checkpoint_average_error,
                                                checkpoint_average_mse],
                                               latest)

                # If we are using the finish when validation stops, we check the current checkpoint
                if g_conf.FINISH_ON_VALIDATION_STALE is not None:
                    if dlib.count_steps_without_decrease(l1_window) > 3 and \
                            dlib.count_steps_without_decrease_robust(l1_window) > 3:
                        coil_logger.write_stop(g_conf.VAL_DATASET_NAME, latest)
                        break

            else:
                latest = get_latest_evaluated_checkpoint()
                time.sleep(1)

                coil_logger.add_message('Loading', {'Message': 'Waiting Checkpoint'})
                print("Waiting for the next Validation")

        print('\n' + 20 * '#')
        print('Finished validation!')
        print(20 * '#')
        coil_logger.add_message('Finished', {})

    except KeyboardInterrupt:
        coil_logger.add_message('Error', {'Message': 'Killed By User'})
        # We erase the output that was unfinished due to some process stop.
        if latest is not None:
            coil_logger.erase_csv(latest)

    except RuntimeError as e:
        if latest is not None:
            coil_logger.erase_csv(latest)
        coil_logger.add_message('Error', {'Message': str(e)})

    except:
        traceback.print_exc()
        coil_logger.add_message('Error', {'Message': 'Something Happened'})
        # We erase the output that was unfinished due to some process stop.
        if latest is not None:
            coil_logger.erase_csv(latest)
