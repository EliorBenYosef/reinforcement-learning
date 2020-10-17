from numpy.random import seed
seed(28)
from tensorflow import set_random_seed
set_random_seed(28)

import tensorflow as tf

from reinforcement_learning.utils.plotter import plot_running_average_comparison
from reinforcement_learning.deep_RL.const import LIBRARY_TF, LIBRARY_KERAS, LIBRARY_TORCH, \
    OPTIMIZER_Adam, OPTIMIZER_RMSprop, OPTIMIZER_Adadelta, OPTIMIZER_Adagrad, OPTIMIZER_SGD
from reinforcement_learning.deep_RL.utils.devices import set_device
from reinforcement_learning.deep_RL.envs import CartPole, Pendulum, MountainCarContinuous, \
    LunarLander, LunarLanderContinuous, BipedalWalker, Breakout, SpaceInvaders

from reinforcement_learning.deep_RL.algorithms.deep_q_learning import Agent, train_agent
# from reinforcement_learning.deep_RL.algorithms.policy_gradient import Agent, train_agent
# from reinforcement_learning.deep_RL.algorithms.actor_critic import Agent, train_agent
# from reinforcement_learning.deep_RL.algorithms.deep_deterministic_policy_gradient import Agent, train_agent


def perform_grid_search(lib_type=LIBRARY_TF, enable_models_saving=False, load_checkpoint=False,
                        perform_random_gameplay=False):

    custom_env = CartPole()
    custom_env.env.seed(28)

    set_device(lib_type, devices_dict=None)

    method_name = 'DQL'

    n_episodes = 3000

    # double_dql = False
    # tau = None

    ###########################################

    # ep_batch_num_list = [1, 10]  # only for PG. 1 = REINFORCE algorithm (MC PG)
    #
    # fc1_dim_list = [64, 128, 256, 512]
    # fc2_dim_list = [64, 128, 256, 512]
    #
    # optimizer_list = [utils.Optimizers.OPTIMIZER_SGD, utils.Optimizers.OPTIMIZER_Adagrad,
    #                   utils.Optimizers.OPTIMIZER_Adadelta, utils.Optimizers.OPTIMIZER_RMSprop,
    #                   utils.Optimizers.OPTIMIZER_Adam]
    # alpha_list = [0.0005, 0.001, 0.002, 0.004]
    # beta_list = [0.0005]  # only for AC & DDPG
    #
    # gamma_list = [0.9, 0.95, 0.99, 1.0]  # [0.9, 0.95, 0.99, 1.0]
    #
    # eps_max_list = [1.0]
    # eps_min_list = [0.0, 0.01, 0.02, 0.04]
    # eps_dec_list = [0.1, 0.2, 0.3, 0.4]  # (1.0 - min) * 2 / n_episodes  # 0.996
    # # eps_dec_type = utils.Calculator.EPS_DEC_LINEAR,
    #
    # memory_size_list = [10000, 100000, 1000000]
    # memory_batch_size_list = [8, 16, 32, 64]

    ###########################################

    # ep_batch_num_list = [1, 10]  # only for PG. 1 = REINFORCE algorithm (MC PG)
    #
    # fc1_dim_list = [256]
    # fc2_dim_list = [256]
    #
    # gamma_list = [0.99]
    # optimizer_list = [utils.Optimizers.OPTIMIZER_Adam]
    # alpha_list = [0.0005]
    # beta_list = [0.0005]  # only for AC & DDPG
    #
    # eps_max_list = [1.0]
    # eps_min_list = [0.1]
    # eps_dec_list = [1.8 / n_episodes]  # (1.0 - min) * 2 / n_episodes  # 0.996
    # # eps_dec_type = utils.Calculator.EPS_DEC_LINEAR,
    #
    # memory_size_list = [1000000]
    # memory_batch_size_list = [64]

    ###########################################

    ep_batch_num_list = [1, 10]  # only for PG. 1 = REINFORCE algorithm (MC PG)

    fc1_dim_list = [64, 128, 256, 512]
    fc2_dim_list = [64, 128, 256, 512]

    optimizer_list = [OPTIMIZER_Adam]
    alpha_list = [0.0005]
    beta_list = [0.0005]  # only for AC & DDPG

    gamma_list = [0.99]

    eps_max_list = [1.0]
    eps_min_list = [0.1]
    eps_dec_list = [1.8 / n_episodes]  # (1.0 - min) * 2 / n_episodes  # 0.996
    # eps_dec_type = utils.Calculator.EPS_DEC_LINEAR,

    memory_size_list = [1000000]
    memory_batch_size_list = [64]

    ###########################################

    counter = 0
    labels = []
    scores_histories_train = []
    scores_histories_test = []

    for optimizer in optimizer_list:
        for alpha in alpha_list:
            for fc1_dim in fc1_dim_list:
                for fc2_dim in fc2_dim_list:
                    for gamma in gamma_list:
                        for eps_max in eps_max_list:
                            for eps_min in eps_min_list:
                                for eps_dec in eps_dec_list:
                                    for memory_size in memory_size_list:
                                        for memory_batch_size in memory_batch_size_list:
                                            counter += 1
                                            print('\n', 'Iteration %d' % counter, '\n')

                                            agent = Agent(
                                                custom_env, [fc1_dim, fc2_dim], n_episodes,
                                                alpha, optimizer_type=optimizer, gamma=gamma,
                                                eps_min=eps_min, eps_dec=eps_dec,  # eps_max=eps_max,
                                                memory_size=memory_size, memory_batch_size=memory_batch_size,
                                                double_dql=False, tau=None, lib_type=lib_type
                                            )

                                            labels.append('[%d,%d]' % (fc1_dim, fc2_dim))

                                            scores_history = train_agent(custom_env, agent, n_episodes,
                                                                         perform_random_gameplay,
                                                                         enable_models_saving, load_checkpoint)
                                            scores_histories_train.append(scores_history)

                                            # scores_history_test = test_trained_agent(
                                            #     custom_env, agent, enable_models_saving)
                                            # scores_histories_test.append(scores_history_test)

                                            tf.reset_default_graph()

    plot_running_average_comparison(
        custom_env.name, scores_histories_train, labels, window=custom_env.window, show=False,
        file_name=custom_env.file_name + '_' + method_name + '_train'
    )
    plot_running_average_comparison(
        custom_env.name, scores_histories_test, labels, window=custom_env.window, show=False,
        file_name=custom_env.file_name + '_' + method_name + '_test'
    )


if __name__ == '__main__':
    perform_grid_search()
