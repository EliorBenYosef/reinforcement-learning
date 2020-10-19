from numpy.random import seed
seed(28)
from tensorflow import set_random_seed
set_random_seed(28)

from reinforcement_learning.utils.plotter import plot_running_average
from reinforcement_learning.deep_RL.const import LIBRARY_TF, LIBRARY_KERAS, LIBRARY_TORCH, \
    OPTIMIZER_Adam, OPTIMIZER_RMSprop, OPTIMIZER_Adadelta, OPTIMIZER_Adagrad, OPTIMIZER_SGD, \
    INPUT_TYPE_OBSERVATION_VECTOR
from reinforcement_learning.deep_RL.utils.utils import get_file_name, test_trained_agent
from reinforcement_learning.deep_RL.utils.devices import set_device
from reinforcement_learning.deep_RL.envs import CartPole, Pendulum, MountainCarContinuous, \
    LunarLander, LunarLanderContinuous, BipedalWalker, Breakout, SpaceInvaders
from reinforcement_learning.deep_RL.algorithms.actor_critic import Agent, train_agent


def play_ac(custom_env, n_episodes, fc_layers_dims, optimizer_type, alpha, beta,
            lib_type=LIBRARY_TORCH, enable_models_saving=False, load_checkpoint=False,
            plot=True, test=False):

    if lib_type == LIBRARY_TF:
        print('\n', "Algorithm currently doesn't work with TensorFlow", '\n')
        return

    # SHARED vs SEPARATE explanation:
    #   SHARED is very helpful in more complex environments (like LunarLander)
    #   you can get away with SEPARATE in less complex environments (like MountainCar)

    if lib_type == LIBRARY_TORCH and custom_env.input_type != INPUT_TYPE_OBSERVATION_VECTOR:
        print('\n', 'the Torch implementation of the Algorithm currently works only with INPUT_TYPE_OBSERVATION_VECTOR!', '\n')
        return

    custom_env.env.seed(28)

    set_device(lib_type, devices_dict=None)

    method_name = 'AC'
    base_dir = 'tmp/' + custom_env.file_name + '/' + method_name + '/'

    agent = Agent(custom_env, fc_layers_dims,
                  optimizer_type, lr_actor=alpha, lr_critic=beta,
                  lib_type=lib_type, base_dir=base_dir)

    scores_history = train_agent(custom_env, agent, n_episodes,
                                 enable_models_saving, load_checkpoint)

    if plot:
        plot_running_average(
            custom_env.name, method_name, scores_history, window=custom_env.window, show=False,
            file_name=get_file_name(custom_env.file_name, agent, n_episodes, method_name) + '_train',
            directory=agent.chkpt_dir if enable_models_saving else None
        )

    scores_history_test = None
    if test:
        scores_history_test = test_trained_agent(custom_env, agent, enable_models_saving)
        if plot:
            plot_running_average(
                custom_env.name, method_name, scores_history_test, window=custom_env.window, show=False,
                file_name=get_file_name(custom_env.file_name, agent, n_episodes, method_name) + '_test',
                directory=agent.chkpt_dir if enable_models_saving else None
            )

    return agent, scores_history, scores_history_test


def test_ac_cartpole():
    custom_env = CartPole()
    fc_layers_dims = [32, 32]
    optimizer_type = OPTIMIZER_Adam
    alpha = 0.0001  # 0.00001
    beta = alpha * 5
    n_episodes = 5  # n_episodes = 2500

    play_ac(custom_env, n_episodes, fc_layers_dims, optimizer_type, alpha, beta, lib_type=LIBRARY_KERAS)


def test_ac_pendulum():
    # custom_env = LunarLander()
    custom_env = Pendulum()

    lib_type = LIBRARY_KERAS

    fc_layers_dims = [2048, 512]  # Keras: [1024, 512]
    optimizer_type = OPTIMIZER_Adam
    alpha = 0.00001
    beta = alpha * 5 if lib_type == LIBRARY_KERAS else None
    n_episodes = 5  # n_episodes = 2000

    play_ac(custom_env, n_episodes, fc_layers_dims, optimizer_type, alpha, beta, lib_type)


def test_ac_mountain_car_continuous():
    custom_env = MountainCarContinuous()
    fc_layers_dims = [256, 256]
    optimizer_type = OPTIMIZER_Adam
    alpha = 0.000005
    beta = alpha * 2
    n_episodes = 5  # n_episodes = 100  # > 100 --> instability (because the value function estimation is unstable)

    play_ac(custom_env, n_episodes, fc_layers_dims, optimizer_type, alpha, beta, lib_type=LIBRARY_KERAS)
