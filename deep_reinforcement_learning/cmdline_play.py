from numpy.random import seed
seed(28)
from tensorflow import set_random_seed
set_random_seed(28)

import argparse

import utils
from deep_reinforcement_learning.envs import Envs

from deep_reinforcement_learning.algorithms.deep_q_learning import Agent, train
# from deep_reinforcement_learning.algorithms.policy_gradient import Agent, train
# from deep_reinforcement_learning.algorithms.actor_critic import Agent, train
# from deep_reinforcement_learning.algorithms.deep_deterministic_policy_gradient import Agent, train


def command_line_play(lib_type=utils.LIBRARY_TF,
                      enable_models_saving=False, load_checkpoint=False,
                      perform_random_gameplay=False):

    # argparse is a built-in module for python that allows us to parse command-line options.
    #   to run it from cmd type: python deep_q_learning.py -n_episodes 100 ...
    #   to run multiple times, put && in between the commands

    # the parser will parse command-line options from cmd text into str \ int\ float \ ...
    parser = argparse.ArgumentParser(description='Command-line Utility for training RL models')

    # The hyphen makes the argument optional (no hyphen makes it a required option)
    #   You can add help='explanation'.

    parser.add_argument('-n_episodes', type=int, default=1)  # 500
    parser.add_argument('-double_dql', type=bool, default=False)  # only for DQL
    parser.add_argument('-tau', type=int, default=None)  # only for DQL
    parser.add_argument('-ep_batch_num', type=int, default=10)  # only for PG. 1 = REINFORCE algorithm (MC PG)

    parser.add_argument('-fc1_dim', type=int, default=256)
    parser.add_argument('-fc2_dim', type=int, default=256)

    parser.add_argument('-optimizer', type=int, default='adam')
    parser.add_argument('-alpha', type=float, default=0.001, help='Learning rate for Optimizer')  # 0.0005, 0.003 ?
    parser.add_argument('-beta', type=float, default=0.0005, help="Learning rate for Critic's optimizer")  # only for AC & DDPG

    parser.add_argument('-gamma', type=float, default=0.99, help='Discount factor for update equation')

    # in epsilon-greedy action selection:
    parser.add_argument('-eps_max', type=float, default=1.0)
    parser.add_argument('-eps_min', type=float, default=0.01)  # EPS_MIN = None
    parser.add_argument('-eps_dec', type=float, default=0.996)  # (max - min) * 2 / episodes
    # eps_dec_type = utils.EPS_DEC_LINEAR,

    parser.add_argument('-memory_size', type=int, default=1000000)
    parser.add_argument('-memory_batch_size', type=int, default=64)

    args = parser.parse_args()

    #####################################

    if args.optimizer == 'sgd':
        optimizer = utils.OPTIMIZER_SGD
    elif args.optimizer == 'adagrad':
        optimizer = utils.OPTIMIZER_Adagrad
    elif args.optimizer == 'adadelta':
        optimizer = utils.OPTIMIZER_Adadelta
    elif args.optimizer == 'rmsprop':
        optimizer = utils.OPTIMIZER_RMSprop
    else:  # args.optimizer == 'adam'
        optimizer = utils.OPTIMIZER_Adam

    #####################################

    custom_env = Envs.ClassicControl.CartPole()
    custom_env.env.seed(28)

    agent = Agent(
        custom_env, [args.fc1_dim, args.fc2_dim], args.n_episodes,
        args.alpha, optimizer_type=optimizer, gamma=args.gamma,
        eps_min=args.eps_min, eps_dec=args.eps_dec,  # eps_max=args.eps_max,
        memory_size=args.memory_size, memory_batch_size=args.memory_batch_size,
        double_dql=args.double_dql, tau=args.tau, lib_type=lib_type
    )

    scores_history = train(custom_env, agent, args.n_episodes, perform_random_gameplay,
                           enable_models_saving, load_checkpoint)

    utils.plot_running_average(
        custom_env.name, scores_history, window=custom_env.window, show=False,
        file_name=utils.get_plot_file_name(custom_env.file_name, agent, memory=True, eps=True)
    )


if __name__ == '__main__':
    command_line_play()
