from numpy.random import seed
seed(28)
from tensorflow.compat.v1 import set_random_seed
set_random_seed(28)

from reinforcement_learning.utils.plotter import plot_running_average
from reinforcement_learning.deep_RL.const import LIBRARY_TF, LIBRARY_KERAS, LIBRARY_TORCH, \
    OPTIMIZER_Adam, OPTIMIZER_RMSprop, OPTIMIZER_Adadelta, OPTIMIZER_Adagrad, OPTIMIZER_SGD
from reinforcement_learning.deep_RL.utils.utils import get_file_name, run_trained_agent
from reinforcement_learning.deep_RL.utils.devices import set_device
from reinforcement_learning.deep_RL.envs import CartPole, Pendulum, MountainCarContinuous, \
    LunarLander, LunarLanderContinuous, BipedalWalker, Breakout, SpaceInvaders
from reinforcement_learning.deep_RL.algorithms.deep_q_learning import Agent, train_agent


def play_dql(custom_env, n_episodes, fc_layers_dims, optimizer_type, alpha, double_dql, tau,
             lib_type=LIBRARY_TF, enable_models_saving=False, load_checkpoint=False,
             perform_random_gameplay=True, rnd_gameplay_episodes=None,
             plot=True, test=False):

    if not custom_env.is_discrete_action_space:
        print('\n', "Environment's Action Space should be discrete!", '\n')
        return

    custom_env.env.seed(28)

    set_device(lib_type, devices_dict=None)

    method_name = 'DQL'
    base_dir = 'tmp/' + custom_env.file_name + '/' + method_name + '/'

    agent = Agent(custom_env, fc_layers_dims, n_episodes,
                  alpha, optimizer_type,
                  double_dql=double_dql, tau=tau, lib_type=lib_type, base_dir=base_dir)

    scores_history = train_agent(custom_env, agent, n_episodes,
                                 enable_models_saving, load_checkpoint,
                                 perform_random_gameplay, rnd_gameplay_episodes)

    if plot:
        plot_running_average(
            custom_env.name, method_name, scores_history,
            # file_name=get_file_name(custom_env.file_name, agent, n_episodes, method_name) + '_train',
            directory=agent.chkpt_dir if enable_models_saving else None
        )

    scores_history_test = None
    if test:
        scores_history_test = run_trained_agent(custom_env, agent, enable_models_saving)
        if plot:
            plot_running_average(
                custom_env.name, method_name, scores_history_test,
                # file_name=get_file_name(custom_env.file_name, agent, n_episodes, method_name) + '_test',
                directory=agent.chkpt_dir if enable_models_saving else None
            )

    return agent, scores_history, scores_history_test


#################################

# Discrete AS:

def run_dql_cartpole(libtype):
    # custom_env = LunarLander()
    custom_env = CartPole()
    fc_layers_dims = [256, 256]
    optimizer_type = OPTIMIZER_Adam
    alpha = 0.0005  # 0.003 ?
    double_dql = False
    tau = None
    n_episodes = 5  # n_episodes = 500  # ~150-200 solves LunarLander

    play_dql(custom_env, n_episodes, fc_layers_dims, optimizer_type, alpha, double_dql, tau, libtype,
             rnd_gameplay_episodes=n_episodes)


def run_dql_lunar_lander(libtype):
    custom_env = LunarLander()
    fc_layers_dims = [256, 256]
    optimizer_type = OPTIMIZER_Adam
    alpha = 0.0005  # 0.003 ?
    double_dql = False
    tau = None
    n_episodes = 5  # n_episodes = 500  # ~150-200 solves LunarLander

    play_dql(custom_env, n_episodes, fc_layers_dims, optimizer_type, alpha, double_dql, tau, libtype,
             rnd_gameplay_episodes=n_episodes)


def run_dql_breakout(libtype):
    custom_env = Breakout()
    fc_layers_dims = [1024]
    optimizer_type = OPTIMIZER_RMSprop  # OPTIMIZER_SGD
    alpha = 0.00025
    double_dql = True
    tau = 10000
    n_episodes = 2  # n_episodes = 200  # start with 200, then 5000 ?

    play_dql(custom_env, n_episodes, fc_layers_dims, optimizer_type, alpha, double_dql, tau, libtype,
             rnd_gameplay_episodes=n_episodes)


def run_dql_space_invaders(libtype):
    custom_env = SpaceInvaders()
    fc_layers_dims = [1024]
    optimizer_type = OPTIMIZER_RMSprop  # OPTIMIZER_SGD
    alpha = 0.003
    double_dql = True
    tau = None
    n_episodes = 2  # n_episodes = 50

    play_dql(custom_env, n_episodes, fc_layers_dims, optimizer_type, alpha, double_dql, tau, libtype,
             rnd_gameplay_episodes=n_episodes)


#################################

def run_test_OBSVEC_DISCRETE(lib_type):
    run_dql_cartpole(lib_type)
    run_dql_lunar_lander(lib_type)


def run_test_FRAMES_DISCRETE(lib_type):
    run_dql_breakout(lib_type)
    run_dql_space_invaders(lib_type)


#################################

def test_OBSVEC_TF():
    run_test_OBSVEC_DISCRETE(LIBRARY_TF)


def test_OBSVEC_KERAS():
    run_test_OBSVEC_DISCRETE(LIBRARY_KERAS)


def test_OBSVEC_TORCH():
    run_test_OBSVEC_DISCRETE(LIBRARY_TORCH)


def test_FRAMES_TF():
    run_test_FRAMES_DISCRETE(LIBRARY_TF)


def test_FRAMES_KERAS():
    run_test_FRAMES_DISCRETE(LIBRARY_KERAS)


def test_FRAMES_TORCH():
    run_test_FRAMES_DISCRETE(LIBRARY_TORCH)
