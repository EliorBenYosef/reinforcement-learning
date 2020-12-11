import numpy as np
import torch

from reinforcement_learning.deep_RL.const import OPTIMIZER_Adam, OPTIMIZER_RMSprop, OPTIMIZER_Adadelta, OPTIMIZER_Adagrad
from reinforcement_learning.utils.utils import run_method, watch_method, pickle_save


# Action Selection:

def eps_greedy(a, EPS, action_space):
    """
    epsilon greedy strategy.
    pure exploration, no chance to get the greedy action.
    """
    rand = np.random.random()
    if rand < EPS:
        modified_action_space = action_space.copy()
        modified_action_space.remove(a)
        a = np.random.choice(modified_action_space)
    return a


def eps_greedy_rnd(a, EPS, action_space):
    """
    epsilon greedy strategy.
    here there's also a chance to get the greedy action.
    """
    rand = np.random.random()
    if rand < EPS:
        a = np.random.choice(action_space)
    return a


class OUActionNoise(object):
    """
    Ornstein-Uhlenbeck Action Noise.
    used for continuous action-space exploration.
    https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py
    """

    def __init__(self, mu, sigma, theta=.15, dt=1e-2, x0=None):
        self.mu = mu
        self.sigma = sigma
        self.theta = theta
        self.dt = dt  # differential with respect to time - this is a temporally correlated noise
        self.x0 = x0  # initial value
        self.reset()  # resets the temporal correlation

    def reset(self):
        """
        sets the previous value for the noise
        """
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)


# Calculator:

def calc_conv_layer_output_dim(Dimension, Filter, Padding, Stride):
    return (Dimension - Filter + 2 * Padding) // Stride + 1


def calc_conv_layer_output_dims(Height, Width, Filter, Padding, Stride):
    h = (Height - Filter + 2 * Padding) // Stride + 1
    w = (Width - Filter + 2 * Padding) // Stride + 1
    return h, w


# Tester:

def run_trained_agent(custom_env, agent, enable_models_saving, episodes=1000):
    total_scores, total_accumulated_scores = run_method(
        custom_env, episodes,
        lambda s: agent.choose_action(s))
    if enable_models_saving:
        pickle_save(total_scores, 'scores_history_test', agent.chkpt_dir)
    return total_scores, total_accumulated_scores


# Watcher:

def watch_trained_agent(custom_env, agent, episodes=3):
    watch_method(custom_env, episodes, lambda s: agent.choose_action(s))


# General:

def torch_compare_nn_params(current_nn, original_nn):
    """
    verify copy
    """
    current_nn_dict = dict(current_nn.named_parameters())
    original_nn_dict = dict(original_nn.named_parameters())
    print("Checking Neural-Network's parameters")
    for param in current_nn_dict:
        print(param, torch.equal(original_nn_dict[param], current_nn_dict[param]))

    input()


def get_file_name(env_file_name, agent, episodes, method_name):
    # options:
    #   .replace('.', 'p')
    #   .split('.')[1]

    if env_file_name is not None:
        env = env_file_name + '_'
    else:
        env = ''

    ############################

    gamma = 'G' + str(agent.GAMMA).replace('.', 'p') + '_'  # 'GAMMA-'

    fc_layers_dims = 'FC-'
    for i, fc_layer_dims in enumerate(agent.fc_layers_dims):
        if i:
            fc_layers_dims += 'x'
        fc_layers_dims += str(fc_layer_dims)
    fc_layers_dims += '_'

    ############################

    optimizer = 'OPT_'
    if agent.optimizer_type == OPTIMIZER_Adam:
        optimizer += 'adam_'
    elif agent.optimizer_type == OPTIMIZER_RMSprop:
        optimizer += 'rms_'  # 'rmsprop_'
    elif agent.optimizer_type == OPTIMIZER_Adadelta:
        optimizer += 'adad_'  # 'adadelta_'
    elif agent.optimizer_type == OPTIMIZER_Adagrad:
        optimizer += 'adag_'  # 'adagrad_'
    else:  # agent.optimizer_type == OPTIMIZER_SGD
        optimizer += 'sgd_'
    alpha = 'a-' + str(agent.ALPHA).replace('.', 'p') + '_'  # 'alpha-'

    if method_name == 'AC' or method_name == 'DDPG':
        beta = 'b-' + str(agent.BETA).replace('.', 'p') + '_'  # 'beta-'
    else:
        beta = ''

    ############################

    if method_name == 'DQL':
        eps_max = 'max-' + str(agent.eps_max).replace('.', 'p') + '_'
        eps_min = 'min-' + str(agent.eps_min).replace('.', 'p') + '_'
        eps_dec = 'dec-' + str(agent.eps_dec).replace('.', 'p') + '_'
        eps = 'EPS_' + eps_max + eps_min + eps_dec
    else:
        eps = ''

    ############################

    if method_name == 'DQL' or method_name == 'DDPG':
        memory_size = 'size-' + str(agent.memory_size)
        memory_batch_size = 'batch-' + str(agent.memory_batch_size)
        replay_buffer = 'MEM_' + memory_size + memory_batch_size
    else:
        replay_buffer = ''

    if method_name == 'PG':
        ep_batch_num = 'PG-ep-batch-' + str(agent.ep_batch_num) + '_'
    else:
        ep_batch_num = ''

    ############################

    episodes = 'N-' + str(episodes)  # n_episodes

    file_name = env + gamma + fc_layers_dims + \
                optimizer + alpha + beta + \
                eps + replay_buffer + ep_batch_num + episodes

    return file_name
