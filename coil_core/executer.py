import os
import time
import multiprocessing
import heapq

from coilutils.experiment_schedule import get_gpu_resources, allocate_gpu_resources, \
    mount_experiment_heap
from coilutils.general import create_exp_path, create_log_folder, erase_wrong_plotting_summaries, erase_validations
from logger import printer, monitorer

from . import train, validate, run_drive


def execute_train(gpu, exp_folder, exp_alias, suppress_output=True, number_of_workers=12):
    """
    Args:
        gpu: The gpu being used for this execution.
        exp_folder: Folder name in configs
        exp_alias: The experiment alias (yaml file)
        suppress_output:
        number_of_workers:
    Returns:
    """
    create_log_folder(exp_folder)
    create_exp_path(exp_folder, exp_alias)
    p = multiprocessing.Process(target=train.execute,
                                args=(gpu, exp_folder, exp_alias, suppress_output, number_of_workers))
    p.start()


def execute_validation(gpu, exp_folder, exp_alias, validation_datasets, erase_bad_validations, restart_validations, suppress_output=True):
    """
    Args:
        gpu: The gpu being used for this execution.
        exp_folder: The folder this driving experiment is being executed
        exp_alias: The experiment alias, file name, to be executed.
        validation_datasets: Validation datasets to be deleted/restarted
        erase_bad_validations:
        restart_validations:
        suppress_output:
    Returns:
    """
    create_log_folder(exp_folder)
    create_exp_path(exp_folder, exp_alias)
    if erase_bad_validations:
        erase_wrong_plotting_summaries(exp_folder, validation_datasets)
    if restart_validations:
        erase_validations(exp_folder, validation_datasets)
    # The difference between train and validation is the
    p = multiprocessing.Process(target=validate.execute,
                                args=(gpu, exp_folder, exp_alias, suppress_output))
    p.start()


def execute_drive(gpu, exp_folder, exp_alias, exp_set_name, suppress_output, docker, record_collisions, no_screen):
    """
    Args:
        gpu: The gpu being used for this execution.
        exp_folder: The folder this driving experiment is being executed
        exp_alias: The experiment alias, file name, to be executed.
        exp_set_name:
        suppress_output:
        docker:
        record_collisions:
        no_screen:
    Returns:
    """
    create_log_folder(exp_folder)
    create_exp_path(exp_folder, exp_alias)
    p = multiprocessing.Process(target=run_drive.execute,
                                args=(gpu, exp_folder, exp_alias, exp_set_name, suppress_output,
                                      docker, record_collisions, no_screen))
    p.start()


# TODO: Finish this or delete it
def folder_execute(exp_folder, exp_set_name, gpus, validation_datasets, driving_environments, is_training,
                   number_of_workers, suppress_output, docker, record_collisions, no_screen,
                   erase_bad_validations, restart_validations):
    """
    Execute a folder of experiments. It will execute trainings and
    all the selected evaluations for each of the models present on the folder.

    Args
        params: a dictionary containing:
            gpus: the gpu numbers that are going  to be allocated for the experiment
            gpu_value: the "value" of each gpu, depending on the value more or less experiments
                        will be allocated per GPU
            folder: the folder where all the experiment configuration files are
            validation_datasets: the validation datasets that are going to be validated
                                 per experiment
            driving_environments: The driving environments where the models are going to be tested.
            record_collisions:
            no_screen:
            erase_bad_validations:
            restart_validations:

    """
    # We set by default that each gpu has a value of 3.5, allowing a training and a driving/validation
    allocation_parameters = {'gpu_value': 3.5, 'train_cost': 1.5, 'validation_cost': 1.0, 'drive_cost': 1.5}

    create_log_folder(exp_folder)
    experiments_list = os.listdir(os.path.join('configs', exp_folder))
    experiments_list = [experiment.split('.')[-2] for experiment in experiments_list]

    allocated_gpus = {gpu: allocation_parameters['gpu_value'] for gpu in gpus}

    executing_processes = []

    free_gpus, resources_on_most_free_gpu, executing_processes = get_gpu_resources(allocated_gpus,
                                                                                   executing_processes,
                                                                                   allocation_parameters)

    # Is a queue of tasks to be executed. The priority is always train, then test, then val.
    tasks_queue = mount_experiment_heap(exp_folder, experiments_list, is_training,
                                        [], [], validation_datasets, driving_environments)

    # No process is executing right now.

    while True:
        #   if not done or executing  get to the list
        # If amount of resources is smaller than a threshold.

        while resources_on_most_free_gpu >= min([allocation_parameters['train_cost'],
                                                 allocation_parameters['validation_cost'],
                                                 allocation_parameters['drive_cost']]) \
                and tasks_queue != []:
            # Allocate all the gpus
            popped_thing = heapq.heappop(tasks_queue)
            process_specs = popped_thing[2]  # To get directly the dict

            # Get the train status, that will affect in scheduling a validation or drive process
            train_status = monitorer.get_status(exp_folder, process_specs['experiment'], 'train')[0]
            # ADD TRAIN TO EXECUTE
            if process_specs['type'] == 'train' and resources_on_most_free_gpu >= \
                    allocation_parameters['train_cost']:
                free_gpus, resources_on_most_free_gpu, gpu_number = allocate_gpu_resources(
                    free_gpus,
                    allocation_parameters['train_cost'])

                execute_train(gpu_number, process_specs['folder'], process_specs['experiment'], number_of_workers)
                process_specs.update({'gpu': gpu_number})

                executing_processes.append(process_specs)
            # ADD DRIVE TO EXECUTE
            elif process_specs['type'] == 'drive' and resources_on_most_free_gpu >= \
                    allocation_parameters['drive_cost'] \
                    and (train_status == 'Iterating' or train_status == 'Loading' or
                         train_status == 'Finished'):
                free_gpus, resources_on_most_free_gpu, gpu_number = allocate_gpu_resources(
                                            free_gpus, allocation_parameters['drive_cost'])

                execute_drive(gpu_number, process_specs['folder'], process_specs['experiment'],
                              process_specs['environment'], suppress_output, docker, record_collisions, no_screen)
                process_specs.update({'gpu': gpu_number})
                executing_processes.append(process_specs)
            # ADD VALIDATION TO EXECUTE
            elif process_specs['type'] == 'validation' and resources_on_most_free_gpu >= \
                    allocation_parameters['validation_cost'] \
                    and (train_status == 'Iterating' or train_status == 'Loading' or
                         train_status == 'Finished'):
                free_gpus, resources_on_most_free_gpu, gpu_number = allocate_gpu_resources(
                                        free_gpus, allocation_parameters['validation_cost'])
                execute_validation(gpu_number, process_specs['folder'], process_specs['experiment'],
                                   erase_bad_validations, restart_validations, suppress_output)
                process_specs.update({'gpu': gpu_number})
                executing_processes.append(process_specs)

        tasks_queue = mount_experiment_heap(exp_folder, experiments_list, is_training,
                                            executing_processes, tasks_queue,
                                            validation_datasets, driving_environments, False)

        printer.plot_folder_summaries(exp_folder,
                                      is_training,
                                      validation_datasets,
                                      driving_environments)
        # Check allocated process, and look which ones finished.

        if len(tasks_queue) == 0 and len(executing_processes) == 0:
            break

        free_gpus, resources_on_most_free_gpu, executing_processes = get_gpu_resources(
            allocated_gpus,
            executing_processes,
            allocation_parameters)

        time.sleep(10)

    print("ALL EXPERIMENTS EXECUTED")
