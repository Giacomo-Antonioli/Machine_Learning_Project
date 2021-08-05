import math

import numpy as np

from Function import Function


def linear_lr_decay(curr_lr, base_lr, final_lr, curr_step, limit_step, **kwargs):
    """
    The linear_lr_decay, linearly decays the learning rate (base_lr) by a decay_rate until iteration "limit_step".
    Then it stops decaying and uses a fix learning rate (final_lr):
    :param curr_lr: current learning (decayed)
    :param base_lr: initial learning rate
    :param final_lr: final (fixed) learning rate
    :param curr_step: current iteration
    :param limit_step: corresponds to the number of step when the learning rate will stop decaying
    :return: updated (decayed) learning rate
    """
    if curr_step < limit_step and curr_lr > final_lr:
        decay_rate = curr_step / limit_step
        curr_lr = (1. - decay_rate) * base_lr + decay_rate * final_lr
        return curr_lr
    return final_lr


def exp_lr_decay(base_lr, decay_rate, curr_step, decay_steps, staircase=False, **kwargs):
    """
    The exp_lr_decay, decays exponentially the learning rate by `decay_rate` every `decay_steps`,
    starting from a `base_lr`
    :param base_lr: The learning rate at the first step
    :param decay_rate: The amount to decay the learning rate at each new stage
    :param decay_steps: The length of each stage, in steps
    :param staircase: If True, only adjusts the learning rate at the stage transitions, producing a step-like decay
    schedule. If False, adjusts the learning rate after each step, creating a smooth decay schedule. Default is True
    :return: updated (decayed) learning rate
    """
    cur_stage = curr_step / decay_steps
    if staircase:
        cur_stage = np.floor(cur_stage)
    decay = -decay_rate * cur_stage
    return base_lr * math.exp(decay)


LinearLRDecay = Function(linear_lr_decay, 'linear')
ExponentialLRDecay = Function(exp_lr_decay, 'exponential')

lr_decays = {
    'linear': LinearLRDecay,
    'exponential': ExponentialLRDecay
}
