"""
https://docs.python.org/2/library/argparse.html#adding-arguments
"""

from tests.test_deep_rl.test_dql import play_dql
from tests.test_deep_rl.test_pg import play_pg
from tests.test_deep_rl.test_ddpg import play_ddpg
from tests.test_deep_rl.test_ac import play_ac

import sys
import argparse

from reinforcement_learning.deep_RL.const import LIBRARY_TF, LIBRARY_KERAS, LIBRARY_TORCH, \
    OPTIMIZER_Adam, OPTIMIZER_RMSprop, OPTIMIZER_Adadelta, OPTIMIZER_Adagrad, OPTIMIZER_SGD
from reinforcement_learning.deep_RL.envs import CartPole, Pendulum, MountainCarContinuous, \
    LunarLander, LunarLanderContinuous, BipedalWalker, Breakout, SpaceInvaders


def parse_args(args):
    """
    Parse arguments from command line input.
    argparse is a built-in module for python that allows us to parse command-line options, from cmd text into variables.
        to run it from cmd, open the terminal or open cmd from the project's root folder, and type the command:
            python -m reinforcement_learning.deep_RL.utils.cmdline_play --lib torch --algo PG --env cp --n 100
            python3 -m reinforcement_learning.deep_RL.utils.cmdline_play --lib torch --algo PG --env cp --n 100
        to run multiple times, put && in between the (full) commands (python ...)

    parser.add_argument('', type=, default=, help='')
    arguments / options:
        Positional (required) arguments - passed without the arg name.
            no dash ('foo') : positional arguments are
        Optional argument - passed with the arg name like so: -arg_name arg_value
            single dash ('-f') : short option /  collection of short options (-abc == -a -b -c)
            double dash ('--foo)' : long option (--abc != -a -b -c)
    kwargs:
        type=bool (default) / str / int / float
        default=                                        # 'true' for bool ??
        action='store_true'                             # default bool
        help='argument description / explanation'
        dest='var_name'                                 # destination variable name

    https://argparsejl.readthedocs.io/en/latest/argparse.html#the-parse-args-function
    """

    parser = argparse.ArgumentParser(description='Training parameters')

    parser.add_argument('--lib', type=str, default='torch', help="Python library type "
                        "{LIBRARY_TF (tf), LIBRARY_KERAS (keras), LIBRARY_TORCH (torch)}")

    parser.add_argument('--algo', type=str, default='AC', help="Algorithm "
                        "{DQL, PG, DDPG, AC}")
    parser.add_argument('--env', type=str, default='si', help="OpenAI Gym Environment "
                        "{CartPole (cp), Pendulum (p), MountainCarContinuous (mcc), LunarLander (ll), "
                        "LunarLanderContinuous (llc), BipedalWalker (bpw), Breakout (bo), SpaceInvaders (si)}")

    parser.add_argument('--n', type=int, default=500, help='Number of training episodes')
    parser.add_argument('--batch', type=int, default=10, help='PG - Episode batch number (1 is REINFORCE = MC PG)')
    parser.add_argument('--mem_s', type=int, default=1000000, help='DQL/DDPG - Memory size for Experience Replay')
    parser.add_argument('--mem_bs', type=int, default=64, help='DQL/DDPG - Memory batch size for Experience Replay')

    parser.add_argument('--fc1', type=int, default=256, help='Dimensions of the first FC layer')
    parser.add_argument('--fc2', type=int, default=256, help='Dimensions of the second FC layer')

    parser.add_argument('--opt', type=str, default='adam', help='Optimizer')
    parser.add_argument('--a', type=float, default=0.001, help='Alpha learning rate for Optimizer')  # 0.0005, 0.003 ?
    parser.add_argument('--b', type=float, default=0.0005, help="DDPG/AC - Beta learning rate for Optimizer")

    # # DQL args:
    #
    # parser.add_argument('--ddql', type=bool, default=False, help='DQL - Perform Double DQL')
    # parser.add_argument('--t', type=int, default=None, help='DQL - Tau value')
    #
    # parser.add_argument('--g', type=float, default=0.99, help='Discount factor for update equation (gamma)')
    #
    # # in epsilon-greedy action selection:
    # parser.add_argument('--eps_max', type=float, default=1.0)
    # parser.add_argument('--eps_min', type=float, default=0.01)  # EPS_MIN = None
    # parser.add_argument('--eps_dec', type=float, default=0.996)  # (max - min) * 2 / episodes
    # # eps_dec_type = utils.Calculator.EPS_DEC_LINEAR,

    # parser.set_defaults(render=False)
    return parser.parse_args(args)


def command_line_play(args=None):
    enable_models_saving = False
    load_checkpoint = False

    if args is None:
        args = sys.argv[1:]  # https://www.pythonforbeginners.com/system/python-sys-argv
    args = parse_args(args)

    if args.env == 'cp':
        custom_env = CartPole()
    elif args.env == 'p':
        custom_env = Pendulum()
    elif args.env == 'mcc':
        custom_env = MountainCarContinuous()
    elif args.env == 'll':
        custom_env = LunarLander()
    elif args.env == 'llc':
        custom_env = LunarLanderContinuous()
    elif args.env == 'bpw':
        custom_env = BipedalWalker()
    elif args.env == 'bo':
        custom_env = Breakout()
    else:  # args.env == 'si'
        custom_env = SpaceInvaders()

    if args.lib == 'tf':
        lib_type = LIBRARY_TF
    elif args.lib == 'keras':
        lib_type = LIBRARY_KERAS
    else:  # args.lib == 'torch'
        lib_type = LIBRARY_TORCH

    fc_layers_dims = [args.fc1, args.fc2]

    if args.opt == 'sgd':
        optimizer_type = OPTIMIZER_SGD
    elif args.opt == 'adagrad':
        optimizer_type = OPTIMIZER_Adagrad
    elif args.opt == 'adadelta':
        optimizer_type = OPTIMIZER_Adadelta
    elif args.opt == 'rmsprop':
        optimizer_type = OPTIMIZER_RMSprop
    else:  # args.opt == 'adam'
        optimizer_type = OPTIMIZER_Adam

    alpha = args.a
    n_episodes = args.n

    if args.algo == 'DQL':
        double_dql = args.ddql
        tau = args.t
        perform_random_gameplay = False
        # gamma = args.g
        # eps_min = args.eps_min
        # eps_dec = args.eps_dec
        # eps_max = args.eps_max
        # memory_size = args.mem_s
        # memory_batch_size = args.mem_bs
        play_dql(custom_env, n_episodes, fc_layers_dims, optimizer_type, alpha, double_dql, tau,
                 lib_type, enable_models_saving, load_checkpoint, perform_random_gameplay)

    elif args.algo == 'PG':
        ep_batch_num = args.batch
        play_pg(custom_env, n_episodes, fc_layers_dims, optimizer_type, alpha, ep_batch_num,
                lib_type, enable_models_saving, load_checkpoint)

    elif args.algo == 'DDPG':
        beta = args.b
        # memory_size = args.mem_s
        # memory_batch_size = args.mem_bs
        play_ddpg(custom_env, n_episodes, fc_layers_dims, optimizer_type, alpha, beta,
                  lib_type, enable_models_saving, load_checkpoint)

    else:  # args.algo == 'AC'
        beta = args.b
        play_ac(custom_env, n_episodes, fc_layers_dims, optimizer_type, alpha, beta,
                lib_type, enable_models_saving, load_checkpoint)


if __name__ == '__main__':
    command_line_play()
