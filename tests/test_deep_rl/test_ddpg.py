from numpy.random import seed
seed(28)
from tensorflow import random
random.set_seed(28)

from reinforcement_learning.utils.plotter import plot_running_average
from reinforcement_learning.deep_RL.const import LIBRARY_TF, LIBRARY_KERAS, LIBRARY_TORCH, \
    OPTIMIZER_Adam, OPTIMIZER_RMSprop, OPTIMIZER_Adadelta, OPTIMIZER_Adagrad, OPTIMIZER_SGD, \
    INPUT_TYPE_OBSERVATION_VECTOR
from reinforcement_learning.deep_RL.utils.utils import get_file_name, run_trained_agent
from reinforcement_learning.deep_RL.utils.devices import set_device
from reinforcement_learning.deep_RL.envs import CartPole, Pendulum, MountainCarContinuous, \
    LunarLander, LunarLanderContinuous, BipedalWalker, Breakout, SpaceInvaders
from reinforcement_learning.deep_RL.algorithms.deep_deterministic_policy_gradient import Agent, train_agent


def play_ddpg(custom_env, n_episodes, fc_layers_dims, optimizer_type, alpha, beta,
              lib_type=LIBRARY_TF, enable_models_saving=False, load_checkpoint=False,
              plot=True, test=False):

    if custom_env.is_discrete_action_space:
        print('\n', "Environment's Action Space should be continuous!", '\n')
        return

    if custom_env.input_type != INPUT_TYPE_OBSERVATION_VECTOR:
        print('\n', 'Algorithm currently works only with INPUT_TYPE_OBSERVATION_VECTOR!', '\n')
        return

    if lib_type == LIBRARY_KERAS:
        print('\n', "Algorithm currently doesn't work with Keras", '\n')
        return

    tau = 0.001

    custom_env.env.seed(28)

    set_device(lib_type, devices_dict=None)

    method_name = 'DDPG'
    base_dir = 'tmp/' + custom_env.file_name + '/' + method_name + '/'

    agent = Agent(custom_env, fc_layers_dims, tau,
                  optimizer_type, alpha, beta,
                  memory_batch_size=custom_env.memory_batch_size, lib_type=lib_type, base_dir=base_dir)

    scores_history = train_agent(custom_env, agent, n_episodes,
                                 enable_models_saving, load_checkpoint)

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

# Continuous AS:

def run_ddpg_pendulum(lib_type):
    custom_env = Pendulum()
    fc_layers_dims = [800, 600]
    optimizer_type = OPTIMIZER_Adam
    alpha = 0.00005
    beta = alpha * 10
    n_episodes = 2  # n_episodes = 1000

    play_ddpg(custom_env, n_episodes, fc_layers_dims, optimizer_type, alpha, beta, lib_type)


def run_ddpg_mountain_car_continuous(lib_type):
    custom_env = MountainCarContinuous()
    fc_layers_dims = [400, 300]
    optimizer_type = OPTIMIZER_Adam
    alpha = 0.000025
    beta = alpha * 10
    n_episodes = 2  # n_episodes = 1000

    play_ddpg(custom_env, n_episodes, fc_layers_dims, optimizer_type, alpha, beta, lib_type)


def run_ddpg_lunar_lander_continuous(lib_type):
    custom_env = LunarLanderContinuous()
    fc_layers_dims = [400, 300]
    optimizer_type = OPTIMIZER_Adam
    alpha = 0.000025
    beta = alpha * 10
    n_episodes = 2  # n_episodes = 1000

    play_ddpg(custom_env, n_episodes, fc_layers_dims, optimizer_type, alpha, beta, lib_type)


def run_ddpg_bipedal_walker(lib_type):
    custom_env = BipedalWalker()
    fc_layers_dims = [400, 300]
    optimizer_type = OPTIMIZER_Adam
    alpha = 0.00005
    beta = alpha * 10
    n_episodes = 2  # n_episodes = 5000

    play_ddpg(custom_env, n_episodes, fc_layers_dims, optimizer_type, alpha, beta, lib_type)


#################################

def run_test_OBSVEC_CONTINUOUS(lib_type):
    run_ddpg_pendulum(lib_type)  # n_actions = 1
    run_ddpg_mountain_car_continuous(lib_type)  # n_actions = 1
    run_ddpg_lunar_lander_continuous(lib_type)  # n_actions = 2
    run_ddpg_bipedal_walker(lib_type)  # n_actions = 4


#################################

def test_OBSVEC_CONTINUOUS_TF():
    run_test_OBSVEC_CONTINUOUS(LIBRARY_TF)


# def test_OBSVEC_CONTINUOUS_KERAS():
#     run_test_OBSVEC_CONTINUOUS(LIBRARY_KERAS)


def test_OBSVEC_CONTINUOUS_TORCH():
    run_test_OBSVEC_CONTINUOUS(LIBRARY_TORCH)
