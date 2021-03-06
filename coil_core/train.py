import os

import random
import time
import traceback
import torch
import torch.optim as optim
import numpy as np

from configs import g_conf, set_type_of_process, merge_with_yaml
from network import CoILModel, Loss, adjust_learning_rate_auto
from input import CoILDataset, Augmenter, select_balancing_strategy
from logger import coil_logger
from coilutils.checkpoint_schedule import is_ready_to_save, get_latest_saved_checkpoint, check_loss_validation_stopped
from coilutils.general import format_time, save_output


# The main function maybe we could call it with a default name
def execute(gpu, exp_folder, exp_alias, suppress_output=False, number_of_workers=12):
    """
        The main training function. This functions loads the latest checkpoint
        for a given, exp_batch (folder) and exp_alias (experiment configuration).
        With this checkpoint it starts from the beginning or continue some training.
    Args:
        gpu: The GPU number
        exp_folder: the folder with the experiments
        exp_alias: the alias, experiment name
        suppress_output: if the output are going to be saved on a file
        number_of_workers: the number of threads used for data loading

    Returns:
        None
    """
    try:
        # We set the visible cuda devices to select the GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu
        g_conf.VARIABLE_WEIGHT = {}
        # At this point the log file with the correct naming is created.
        # You merge the yaml file with the global configuration structure.
        merge_with_yaml(os.path.join('configs', exp_folder, f'{exp_alias}.yaml'))
        set_type_of_process('train')
        # Set the process into loading status.
        coil_logger.add_message('Loading', {'GPU': os.environ['CUDA_VISIBLE_DEVICES']})

        # Put the output to a separate file if it is the case
        if suppress_output:
            save_output(exp_alias)

        if coil_logger.check_finish('train'):
            coil_logger.add_message('Finished', {})
            return

        # Preload option
        if g_conf.PRELOAD_MODEL_ALIAS is not None:
            checkpoint = torch.load(os.path.join('_logs',
                                                 g_conf.PRELOAD_MODEL_BATCH,
                                                 g_conf.PRELOAD_MODEL_ALIAS,
                                                 'checkpoints',
                                                 f'{g_conf.PRELOAD_MODEL_CHECKPOINT}.pth'))

        # Get the latest checkpoint to be loaded
        # returns none if there are no checkpoints saved for this model
        checkpoint_file = get_latest_saved_checkpoint()
        if checkpoint_file is not None:
            print(f'=>Starting from previous best checkpoint...{get_latest_saved_checkpoint()}')
            checkpoint = torch.load(os.path.join('_logs', exp_folder, exp_alias,
                                                 'checkpoints', str(get_latest_saved_checkpoint())))
            iteration = checkpoint['iteration']
            best_loss = checkpoint['best_loss']
            best_loss_iter = checkpoint['best_loss_iter']
        else:
            print('=>Starting from scratch...')
            iteration = 0
            best_loss = 10000.0
            best_loss_iter = 0

        # Define the dataset. This structure is has the __get_item__ redefined in a way
        # that you can access the positions from the root directory as a in a vector.
        full_dataset = os.path.join(os.environ["COIL_DATASET_PATH"], g_conf.TRAIN_DATASET_NAME)

        # By instantiating the augmenter we get a callable that augment images and transform them
        # into tensors.
        augmenter = Augmenter(g_conf.AUGMENTATION)

        # Instantiate the class used to read a dataset. The coil dataset generator
        # can be found
        dataset = CoILDataset(full_dataset,
                              transform=augmenter,
                              preload_name=g_conf.TRAIN_DATASET_NAME,
                              process_type='train')
        print("=>Loaded dataset")

        data_loader = select_balancing_strategy(dataset, iteration, number_of_workers)

        model = CoILModel(g_conf.MODEL_TYPE, g_conf.MODEL_CONFIGURATION).cuda()

        optimizer = optim.Adam(model.parameters(), lr=g_conf.LEARNING_RATE)

        if checkpoint_file is not None or g_conf.PRELOAD_MODEL_ALIAS is not None:
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            accumulated_time = checkpoint['total_time']
            loss_window = coil_logger.recover_loss_window('train', iteration)
        else:  # We accumulate iteration time and keep the average speed
            accumulated_time = 0
            loss_window = []

        print(f"=>Setting the loss ({g_conf.LOSS_FUNCTION})")
        criterion = Loss(g_conf.LOSS_FUNCTION)

        # For printing on the console purposes
        iteration_digits = int(np.log10(g_conf.NUMBER_ITERATIONS)) + 1

        print(20 * '#')
        print(' Starting training!')
        print(20 * '#')

        # Loss time series window
        for data in data_loader:
            # Basically in this mode of execution, we validate every X Steps, if it goes up 3 times,
            # add a stop on the _logs folder that is going to be read by this process
            if g_conf.FINISH_ON_VALIDATION_STALE is not None and \
                    check_loss_validation_stopped(iteration, g_conf.FINISH_ON_VALIDATION_STALE):
                break
            """     
                ####################################
                    Main optimization loop
                ####################################
            """
            iteration += 1

            if iteration % 1000 == 0:
                adjust_learning_rate_auto(optimizer, loss_window, coil_logger)

            capture_time = time.time()

            # get the control commands from float_data, size = [g_conf.BATCH_SIZE, len(g_conf.INPUTS)]
            controls = data['directions'].cuda()
            # The output(branches) is a list of 5 branches results, each branch is with size
            # [g_conf.BATCH_SIZE, len(g_conf.TARGETS)]
            model.zero_grad()
            branches = model(torch.squeeze(data['rgb'].cuda()), dataset.extract_inputs(data).cuda())
            loss_function_params = {
                'branches': branches,
                'targets': dataset.extract_targets(data).cuda(),
                'controls': controls,
                'inputs': dataset.extract_inputs(data).cuda(),
                'branch_weights': g_conf.BRANCH_LOSS_WEIGHT,
                'variable_weights': g_conf.VARIABLE_WEIGHT
            }
            loss, _ = criterion(loss_function_params)
            loss.backward()
            optimizer.step()
            """
                ################################################
                Adding tensorboard logs.
                Making calculations for logging purposes.
                These logs are monitored by the printer module.
                #################################################
            """
            coil_logger.add_scalar(tag=f'Training Loss ({g_conf.LOSS_FUNCTION})', value=loss.data, iteration=iteration)
            coil_logger.add_image(tag='Image', images=torch.squeeze(data['rgb']), iteration=iteration)
            if loss.data < best_loss:
                best_loss = loss.data.tolist()
                best_loss_iter = iteration

            # Log a random position
            position = random.randint(0, len(data) - 1)

            output = model.extract_branch(torch.stack(branches[0:4]), controls)
            error = torch.abs(output - dataset.extract_targets(data).cuda())[position].data.tolist()

            accumulated_time += time.time() - capture_time
            coil_logger.add_scalar(tag='Error steer', value=error[0], iteration=iteration)
            coil_logger.add_scalar(tag='Error throttle', value=error[1],  iteration=iteration)
            coil_logger.add_scalar(tag='Error brake', value=error[2], iteration=iteration)
            coil_logger.add_message(phase='Iterating',
                                    message={'Iteration': iteration,
                                             f'Loss ({g_conf.LOSS_FUNCTION})': loss.data.tolist(),
                                             'Images/s': (iteration * g_conf.BATCH_SIZE) / accumulated_time,
                                             'BestLoss': best_loss, 'BestLossIteration': best_loss_iter,
                                             'Output': output[position].data.tolist(),
                                             'GroundTruth': dataset.extract_targets(data)[position].data.tolist(),
                                             'Error': error,
                                             'Inputs': dataset.extract_inputs(data)[position].data.tolist()},
                                    iteration=iteration)
            loss_window.append(loss.data.tolist())
            coil_logger.write_on_error_csv(error_file_name='train', data=loss.data, iteration=iteration)
            """
            ######################################
                        Saving the model 
            ######################################
            """
            # Save the model according to g_conf.SAVE_SCHEDULE
            if is_ready_to_save(iteration):
                state = {
                    'iteration': iteration,
                    'state_dict': model.state_dict(),
                    'best_loss': best_loss,
                    'total_time': accumulated_time,
                    'optimizer': optimizer.state_dict(),
                    'best_loss_iter': best_loss_iter
                }
                torch.save(state, os.path.join('_logs', exp_folder, exp_alias, 'checkpoints', f'{iteration}.pth'))
            # Console message to print (will be on the same line, so we add \r; rest of info can be found in tensorboard
            console_message = f"\r[{iteration:{iteration_digits}d}/{g_conf.NUMBER_ITERATIONS}] - Time: " \
                              f"{format_time(accumulated_time):15s} - Loss: {loss.data:.16f} - Best Loss: "\
                              f"{best_loss:.16f} / Best Loss Iteration: {best_loss_iter}"
            print(console_message, end='')

        print(20*'#')
        print(' Finished training!')
        print(20 * '#')
        coil_logger.add_message('Finished', {})

    except KeyboardInterrupt:
        coil_logger.add_message('Error', {'Message': 'Killed By User'})

    except RuntimeError as e:

        coil_logger.add_message('Error', {'Message': str(e)})

    except:
        traceback.print_exc()
        coil_logger.add_message('Error', {'Message': 'Something Happened'})
