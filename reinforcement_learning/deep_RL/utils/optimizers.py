import tensorflow as tf
import torch
from keras import optimizers as keras_opt
from torch.optim import adagrad as torch_opt_adagrad, adadelta as torch_opt_adadelta, rmsprop as torch_opt_rmsprop

from reinforcement_learning.deep_RL.const import OPTIMIZER_SGD, OPTIMIZER_Adagrad, OPTIMIZER_Adadelta, OPTIMIZER_RMSprop


def tf_get_optimizer(optimizer_type, lr, momentum=0.):  # momentum=0.9
    if optimizer_type == OPTIMIZER_SGD:
        if momentum == 0:
            return tf.compat.v1.train.GradientDescentOptimizer(lr)
        else:
            return tf.compat.v1.train.MomentumOptimizer(lr, momentum)
    elif optimizer_type == OPTIMIZER_Adagrad:
        return tf.compat.v1.train.AdagradOptimizer(lr)
    elif optimizer_type == OPTIMIZER_Adadelta:
        return tf.compat.v1.train.AdadeltaOptimizer(lr)
    elif optimizer_type == OPTIMIZER_RMSprop:
        return tf.compat.v1.train.RMSPropOptimizer(lr, momentum=momentum)
        # return tf.compat.v1.train.RMSPropOptimizer(lr, decay=0.99, momentum=momentum, epsilon=1e-6)
    else:  # optimizer_type == OPTIMIZER_Adam
        return tf.compat.v1.train.AdamOptimizer(lr)


def keras_get_optimizer(optimizer_type, lr, momentum=0., rho=None, epsilon=None, decay=0., beta_1=0.9, beta_2=0.999):
    if optimizer_type == OPTIMIZER_SGD:
        return keras_opt.SGD(lr, momentum, decay)  # momentum=0.9
    elif optimizer_type == OPTIMIZER_Adagrad:
        return keras_opt.Adagrad(lr, epsilon, decay)
    elif optimizer_type == OPTIMIZER_Adadelta:
        return keras_opt.Adadelta(lr, rho if rho is not None else 0.95, epsilon, decay)
    elif optimizer_type == OPTIMIZER_RMSprop:
        return keras_opt.RMSprop(lr, rho if rho is not None else 0.9, epsilon, decay)  # momentum= ?
        # return optimizers.RMSprop(lr, rho=0.99, epsilon=0.1)
        # return optimizers.RMSprop(lr, epsilon=1e-6, decay=0.99)
    else:  # optimizer_type == OPTIMIZER_Adam
        return keras_opt.Adam(lr, beta_1, beta_2, epsilon, decay)


def torch_get_optimizer(optimizer_type, lr, params, momentum=0., weight_decay=0.):  # momentum=0.9
    if optimizer_type == OPTIMIZER_SGD:
        return torch.optim.SGD(params, lr, momentum, weight_decay=weight_decay)
    elif optimizer_type == OPTIMIZER_Adagrad:
        return torch_opt_adagrad.Adagrad(params, lr, weight_decay=weight_decay)
    elif optimizer_type == OPTIMIZER_Adadelta:
        return torch_opt_adadelta.Adadelta(params, lr, weight_decay=weight_decay)
    elif optimizer_type == OPTIMIZER_RMSprop:
        return torch_opt_rmsprop.RMSprop(params, lr, weight_decay=weight_decay, momentum=momentum)
        # return torch_opt_rmsprop.RMSprop(params, lr, eps=1e-6, weight_decay=0.99, momentum=momentum)
    else:  # optimizer_type == OPTIMIZER_Adam
        return torch.optim.Adam(params, lr, weight_decay=weight_decay)
