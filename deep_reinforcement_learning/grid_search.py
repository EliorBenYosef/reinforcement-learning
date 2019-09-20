from numpy.random import seed
seed(28)
from tensorflow import set_random_seed
set_random_seed(28)

import tensorflow as tf

import utils
from deep_reinforcement_learning.envs import Envs

from deep_reinforcement_learning.algorithms.deep_q_learning import Agent, train
# from deep_reinforcement_learning.algorithms.policy_gradient import Agent, train
# from deep_reinforcement_learning.algorithms.actor_critic import Agent, train
# from deep_reinforcement_learning.algorithms.deep_deterministic_policy_gradient import Agent, train


def perform_grid_search(lib_type=utils.LIBRARY_TF, enable_models_saving=False, load_checkpoint=False,
                        perform_random_gameplay=False):

    custom_env = Envs.ClassicControl.CartPole()
    custom_env.env.seed(28)

    # utils.set_device(lib_type)

    n_episodes = 3000

    # double_dql = False
    # tau = None

    ###########################################

    # ep_batch_num_list = [1, 10]  # only for PG. 1 = REINFORCE algorithm (MC PG)
    #
    # fc1_dim_list = [64, 128, 256, 512]
    # fc2_dim_list = [64, 128, 256, 512]
    #
    # optimizer_list = [utils.OPTIMIZER_SGD, utils.OPTIMIZER_Adagrad, utils.OPTIMIZER_Adadelta, utils.OPTIMIZER_RMSprop,
    #                   utils.OPTIMIZER_Adam]
    # alpha_list = [0.0005, 0.001, 0.002, 0.004]
    # beta_list = [0.0005]  # only for AC & DDPG
    #
    # gamma_list = [0.9, 0.95, 0.99, 1.0]  # [0.9, 0.95, 0.99, 1.0]
    #
    # eps_max_list = [1.0]
    # eps_min_list = [0.0, 0.01, 0.02, 0.04]
    # eps_dec_list = [0.1, 0.2, 0.3, 0.4]  # (1.0 - min) * 2 / n_episodes  # 0.996
    # # eps_dec_type = utils.EPS_DEC_LINEAR,
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
    # optimizer_list = [utils.OPTIMIZER_Adam]
    # alpha_list = [0.0005]
    # beta_list = [0.0005]  # only for AC & DDPG
    #
    # eps_max_list = [1.0]
    # eps_min_list = [0.1]
    # eps_dec_list = [1.8 / n_episodes]  # (1.0 - min) * 2 / n_episodes  # 0.996
    # # eps_dec_type = utils.EPS_DEC_LINEAR,
    #
    # memory_size_list = [1000000]
    # memory_batch_size_list = [64]

    ###########################################

    ep_batch_num_list = [1, 10]  # only for PG. 1 = REINFORCE algorithm (MC PG)

    fc1_dim_list = [64, 128, 256, 512]
    fc2_dim_list = [64, 128, 256, 512]

    optimizer_list = [utils.OPTIMIZER_Adam]
    alpha_list = [0.0005]
    beta_list = [0.0005]  # only for AC & DDPG

    gamma_list = [0.99]

    eps_max_list = [1.0]
    eps_min_list = [0.1]
    eps_dec_list = [1.8 / n_episodes]  # (1.0 - min) * 2 / n_episodes  # 0.996
    # eps_dec_type = utils.EPS_DEC_LINEAR,

    memory_size_list = [1000000]
    memory_batch_size_list = [64]

    ###########################################

    counter = 0
    scores_histories = []
    labels = []

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

                                            scores_history = train(
                                                custom_env, agent, n_episodes, perform_random_gameplay,
                                                enable_models_saving, load_checkpoint)
                                            scores_histories.append(scores_history)
                                            labels.append('[%d,%d]' % (fc1_dim, fc2_dim))

                                            tf.reset_default_graph()

    utils.plot_running_average_comparison(
        custom_env.name, scores_histories, labels, window=custom_env.window, show=False,
        file_name=custom_env.file_name + '_dql'
    )


if __name__ == '__main__':
    perform_grid_search()
