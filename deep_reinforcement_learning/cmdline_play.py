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


def command_line_play():

    # argparse is a built-in module for python that allows us to parse command-line options.
    #   to run it from cmd type: python cmdline_play.py -n 100 ...
    #       positional arguments are passed without the arg name
    #       optional arguments are passed with the arg name like so: -arg_name arg_value
    #   to run multiple times, put && in between the commands (python ...)

    # the parser will parse command-line options from cmd text into str \ int\ float \ ...
    parser = argparse.ArgumentParser(description='Command-line Utility for training RL models')

    # The hyphen makes the argument optional (no hyphen makes it a required option)
    #   You can add help='explanation'.

    parser.add_argument('-n', type=int, default=1, help='Number of episodes')  # 500
    parser.add_argument('-ddql', type=bool, default=False, help='Perform Double DQL')
    parser.add_argument('-t', type=int, default=None, help='Tau value for DQL')
    parser.add_argument('-batch', type=int, default=10, help='Episode batch number fo PG (1 is REINFORCE = MC PG)')

    parser.add_argument('-fc1', type=int, default=256, help='Dimensions of the first FC layer')
    parser.add_argument('-fc2', type=int, default=256, help='Dimensions of the second FC layer')

    parser.add_argument('-opt', type=int, default='adam', help='Optimizer')
    parser.add_argument('-a', type=float, default=0.001, help='Learning rate for Optimizer (alpha)')  # 0.0005, 0.003 ?
    parser.add_argument('-b', type=float, default=0.0005, help="Learning rate for Critic's optimizer (beta)")  # only for AC & DDPG

    parser.add_argument('-g', type=float, default=0.99, help='Discount factor for update equation (gamma)')

    # in epsilon-greedy action selection:
    parser.add_argument('-eps_max', type=float, default=1.0)
    parser.add_argument('-eps_min', type=float, default=0.01)  # EPS_MIN = None
    parser.add_argument('-eps_dec', type=float, default=0.996)  # (max - min) * 2 / episodes
    # eps_dec_type = utils.EPS_DEC_LINEAR,

    parser.add_argument('-mem_s', type=int, default=1000000, help='Memory size')
    parser.add_argument('-mem_bs', type=int, default=64, help='Memory batch size')

    args = parser.parse_args()

    #####################################

    if args.opt == 'sgd':
        optimizer = utils.OPTIMIZER_SGD
    elif args.opt == 'adagrad':
        optimizer = utils.OPTIMIZER_Adagrad
    elif args.opt == 'adadelta':
        optimizer = utils.OPTIMIZER_Adadelta
    elif args.opt == 'rmsprop':
        optimizer = utils.OPTIMIZER_RMSprop
    else:  # args.opt == 'adam'
        optimizer = utils.OPTIMIZER_Adam

    #####################################

    method_name = 'AC'
    custom_env = Envs.ClassicControl.CartPole()
    custom_env.env.seed(28)

    lib_type = utils.LIBRARY_TF
    enable_models_saving = False
    load_checkpoint = False
    perform_random_gameplay = False

    if lib_type == utils.LIBRARY_TF:
        utils.tf_set_device()

    agent = Agent(
        custom_env, [args.fc1, args.fc2], args.n,
        args.a, optimizer_type=optimizer, gamma=args.g,
        eps_min=args.eps_min, eps_dec=args.eps_dec,  # eps_max=args.eps_max,
        memory_size=args.mem_s, memory_batch_size=args.mem_bs,
        double_dql=args.ddql, tau=args.t, lib_type=lib_type
    )

    scores_history = train(custom_env, agent, args.n, perform_random_gameplay,
                           enable_models_saving, load_checkpoint)

    utils.plot_running_average(
        custom_env.name, method_name, scores_history, window=custom_env.window, show=False,
        file_name=utils.get_file_name(custom_env.file_name, agent, replay_buffer=True, eps=True),
        directory=agent.chkpt_dir if enable_models_saving else None
    )


if __name__ == '__main__':
    command_line_play()
