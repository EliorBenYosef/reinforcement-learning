from numpy.random import seed
seed(28)
from tensorflow.compat.v1 import set_random_seed
set_random_seed(28)

from reinforcement_learning.utils.plotter import plot_running_average
from reinforcement_learning.deep_RL.const import LIBRARY_TF, LIBRARY_KERAS, LIBRARY_TORCH, \
    OPTIMIZER_Adam, OPTIMIZER_RMSprop, OPTIMIZER_Adadelta, OPTIMIZER_Adagrad, OPTIMIZER_SGD, \
    INPUT_TYPE_OBSERVATION_VECTOR, INPUT_TYPE_STACKED_FRAMES, ATARI_FRAMES_STACK_SIZE, \
    NETWORK_TYPE_SEPARATE, NETWORK_TYPE_SHARED
from reinforcement_learning.deep_RL.utils.utils import get_file_name, run_trained_agent
from reinforcement_learning.deep_RL.utils.devices import set_device
from reinforcement_learning.deep_RL.envs import CartPole, Pendulum, MountainCarContinuous, \
    LunarLander, LunarLanderContinuous, BipedalWalker, Breakout, SpaceInvaders
from reinforcement_learning.deep_RL.algorithms.actor_critic import Agent, train_agent


def play_ac(custom_env, n_episodes, fc_layers_dims, network_type, optimizer_type, alpha, beta,
            lib_type=LIBRARY_TORCH, enable_models_saving=False, load_checkpoint=False,
            plot=True, test=False):
    """
    :param network_type:
        NETWORK_TYPE_SHARED - very helpful in more complex environments (like LunarLander)
        NETWORK_TYPE_SEPARATE - suitable in less complex environments (like MountainCar)
    """

    custom_env.env.seed(28)

    set_device(lib_type, devices_dict=None)

    method_name = 'AC'
    base_dir = 'tmp/' + custom_env.file_name + '/' + method_name + '/'

    agent = Agent(custom_env, fc_layers_dims, network_type,
                  optimizer_type, lr_actor=alpha, lr_critic=beta,
                  lib_type=lib_type, base_dir=base_dir)

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

# Discrete AS:

def run_ac_cartpole(lib_type, network_type):
    custom_env = CartPole()
    fc_layers_dims = [32, 32]
    optimizer_type = OPTIMIZER_Adam
    alpha = 0.0001  # 0.00001
    beta = alpha * 5
    n_episodes = 2  # n_episodes = 2500

    play_ac(custom_env, n_episodes, fc_layers_dims, network_type, optimizer_type, alpha, beta, lib_type)


def run_ac_lunar_lander(lib_type, network_type):
    custom_env = LunarLander()

    fc_layers_dims = [2048, 512]  # Keras: [1024, 512]
    optimizer_type = OPTIMIZER_Adam
    alpha = 0.00001
    beta = alpha * 5
    n_episodes = 2  # n_episodes = 2000

    play_ac(custom_env, n_episodes, fc_layers_dims, network_type, optimizer_type, alpha, beta, lib_type)


def run_ac_breakout(libtype, network_type):
    custom_env = Breakout()
    fc_layers_dims = [1024]
    optimizer_type = OPTIMIZER_RMSprop  # OPTIMIZER_SGD
    # alpha = 0.00025
    alpha = 0.000005
    beta = alpha * 2
    n_episodes = 2  # n_episodes = 200  # start with 200, then 5000 ?

    play_ac(custom_env, n_episodes, fc_layers_dims, network_type, optimizer_type, alpha, beta, libtype)


def run_ac_space_invaders(libtype, network_type):
    custom_env = SpaceInvaders()
    fc_layers_dims = [1024]
    optimizer_type = OPTIMIZER_RMSprop  # OPTIMIZER_SGD
    # alpha = 0.003
    alpha = 0.000005
    beta = alpha * 2
    n_episodes = 2  # n_episodes = 50

    play_ac(custom_env, n_episodes, fc_layers_dims, network_type, optimizer_type, alpha, beta, libtype)


#################################

# Continuous AS:

def run_ac_pendulum(lib_type, network_type):
    custom_env = Pendulum()
    fc_layers_dims = [2048, 512]  # Keras: [1024, 512]
    optimizer_type = OPTIMIZER_Adam
    alpha = 0.00001
    beta = alpha * 5
    n_episodes = 2  # n_episodes = 2000

    play_ac(custom_env, n_episodes, fc_layers_dims, network_type, optimizer_type, alpha, beta, lib_type)


def run_ac_mountain_car_continuous(lib_type, network_type):
    custom_env = MountainCarContinuous()
    fc_layers_dims = [512, 512]
    optimizer_type = OPTIMIZER_Adam
    alpha = 0.00001
    beta = alpha * 2
    n_episodes = 2  # n_episodes = 100  # > 100 --> instability (because the value function estimation is unstable)

    play_ac(custom_env, n_episodes, fc_layers_dims, network_type, optimizer_type, alpha, beta, lib_type)


def run_ac_lunar_lander_continuous(lib_type, network_type):
    custom_env = LunarLanderContinuous()
    fc_layers_dims = [400, 300]
    optimizer_type = OPTIMIZER_Adam
    alpha = 0.000025
    beta = alpha * 10
    n_episodes = 2  # n_episodes = 1000

    play_ac(custom_env, n_episodes, fc_layers_dims, network_type, optimizer_type, alpha, beta, lib_type)


def run_ac_bipedal_walker(lib_type, network_type):
    custom_env = BipedalWalker()
    fc_layers_dims = [400, 300]
    optimizer_type = OPTIMIZER_Adam
    alpha = 0.00005
    beta = alpha * 10
    n_episodes = 2  # n_episodes = 5000

    play_ac(custom_env, n_episodes, fc_layers_dims, network_type, optimizer_type, alpha, beta, lib_type)


#################################

def run_test_OBSVEC_DISCRETE(lib_type):
    run_ac_cartpole(lib_type, network_type=NETWORK_TYPE_SEPARATE)
    run_ac_lunar_lander(lib_type, network_type=NETWORK_TYPE_SHARED)


def run_test_OBSVEC_CONTINUOUS(lib_type):
    run_ac_pendulum(lib_type, network_type=NETWORK_TYPE_SEPARATE)  # n_actions = 1
    # run_ac_mountain_car_continuous(lib_type, network_type=NETWORK_TYPE_SHARED)  # n_actions = 1  # takes too long...
    run_ac_lunar_lander_continuous(lib_type, network_type=NETWORK_TYPE_SEPARATE)  # n_actions = 2
    run_ac_bipedal_walker(lib_type, network_type=NETWORK_TYPE_SHARED)  # n_actions = 4


def run_test_FRAMES_DISCRETE(lib_type):
    run_ac_breakout(lib_type, network_type=NETWORK_TYPE_SEPARATE)
    run_ac_space_invaders(lib_type, network_type=NETWORK_TYPE_SHARED)


#################################

def test_OBSVEC_DISCRETE_TF():
    run_test_OBSVEC_DISCRETE(LIBRARY_TF)


def test_OBSVEC_DISCRETE_KERAS():
    run_test_OBSVEC_DISCRETE(LIBRARY_KERAS)


def test_OBSVEC_DISCRETE_TORCH():
    run_test_OBSVEC_DISCRETE(LIBRARY_TORCH)


def test_OBSVEC_CONTINUOUS_TF():
    run_test_OBSVEC_CONTINUOUS(LIBRARY_TF)


def test_OBSVEC_CONTINUOUS_KERAS():
    run_test_OBSVEC_CONTINUOUS(LIBRARY_KERAS)


def test_OBSVEC_CONTINUOUS_TORCH():
    run_test_OBSVEC_CONTINUOUS(LIBRARY_TORCH)


def test_FRAMES_DISCRETE_TF():
    run_test_FRAMES_DISCRETE(LIBRARY_TF)


def test_FRAMES_DISCRETE_KERAS():
    run_test_FRAMES_DISCRETE(LIBRARY_KERAS)


def test_FRAMES_DISCRETE_TORCH():
    run_test_FRAMES_DISCRETE(LIBRARY_TORCH)
