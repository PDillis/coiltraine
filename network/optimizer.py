import dlib

from configs import g_conf


def adjust_learning_rate(optimizer, num_iters, scheduler='normal'):
    """
    Adjusts the learning rate every epoch based on the selected schedule
    """
    cur_iters = num_iters
    minlr = 0.0000001
    learning_rate = g_conf.LEARNING_RATE
    decayinterval = g_conf.LEARNING_RATE_DECAY_INTERVAL
    decaylevel = g_conf.LEARNING_RATE_DECAY_LEVEL
    if scheduler == "normal":
        while cur_iters >= decayinterval:
            learning_rate = learning_rate * decaylevel
            cur_iters = cur_iters - decayinterval
        learning_rate = max(learning_rate, minlr)

    for param_group in optimizer.param_groups:
        print(f"Learning rate is {learning_rate}")
        param_group['lr'] = learning_rate


def adjust_learning_rate_auto(optimizer, loss_window, coil_logger):
    """
    Adjusts the learning rate every epoch based on the selected schedule
    """
    minlr = 0.0000001
    learning_rate = g_conf.LEARNING_RATE
    n = 1000
    start_point = 0
    while n < len(loss_window):
        # use dlib.net to see for how many steps the noisy loss has gone without noticeably decreasing in value
        steps_no_decrease = dlib.count_steps_without_decrease(loss_window[start_point:n])
        # ibidem, just discarding the 10% largest values
        steps_no_decrease_robust = dlib.count_steps_without_decrease_robust(loss_window[start_point:n])
        coil_logger.add_message('{(Start_point/n): (Steps not decreased/not decreased robust)}',
                                {f'({start_point}/{n})': f'{steps_no_decrease}/{steps_no_decrease_robust})'})
        if steps_no_decrease > g_conf.LEARNING_RATE_THRESHOLD and \
                steps_no_decrease_robust > g_conf.LEARNING_RATE_THRESHOLD:
            start_point = n
            learning_rate = learning_rate * g_conf.LEARNING_RATE_DECAY_LEVEL

        n += 1000

    learning_rate = max(learning_rate, minlr)
    coil_logger.add_message('Learning rate', {'lr': learning_rate})

    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate
