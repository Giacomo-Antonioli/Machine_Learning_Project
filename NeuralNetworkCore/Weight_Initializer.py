import numpy as np


def weights_initializers(init_type, **kwargs):
    inits = {
        'constant': _constant,
        'glorot_uniform': _glorot_uniform,
        'glorot_normal': _glorot_normal,
        'he_uniform': _he_uniform,
        'he_normal': _he_normal,
        'uniform': _rand_init
    }
    return inits[init_type](**kwargs)


def _rand_init(fan_in, fan_out, limits=(-0.1, 0.1)):
    lower_lim, upper_lim = limits[0], limits[1]
    if lower_lim >= upper_lim:
        raise ValueError(f"lower_lim must be <= than upper_lim")
    res = np.random.uniform(low=lower_lim, high=upper_lim, size=(fan_in, fan_out))
    if fan_in == 1:
        return res[0]
    return res


def _constant(fan_in, fan_out, init_value):
    if fan_in == 1:
        return np.full(shape=fan_out, fill_value=init_value)
    return np.full(shape=(fan_in, fan_out), fill_value=init_value)


def _glorot_uniform(fan_in, fan_out):
    sd = np.sqrt(6.0 / (fan_in + fan_out))
    res = np.random.uniform(low=-sd, high=sd, size=(fan_in, fan_out))
    if fan_in == 1:
        return res[0]
    return res


def _glorot_normal(fan_in, fan_out):
    sd = np.sqrt(2.0 / (fan_in + fan_out))
    res = np.random.normal(loc=0.0, scale=sd, size=(fan_in, fan_out))
    if fan_in == 1:
        return res[0]
    return res


def _he_uniform(fan_in, fan_out):
    sd = np.sqrt(6.0 / (fan_in))
    res = np.random.uniform(low=-sd, high=sd, size=(fan_in, fan_out))
    if fan_in == 1:
        return res[0]
    return res


def _he_normal(fan_in, fan_out):
    sd = np.sqrt(2.0 / (fan_in))
    res = np.random.normal(loc=0.0, scale=sd, size=(fan_in, fan_out))
    if fan_in == 1:
        return res[0]
    return res
