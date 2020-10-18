from tests.test_deep_rl.test_dql import play_dql
# from tests.test_deep_rl.test_pg import play_pg
# from tests.test_deep_rl.test_ddpg import play_ddpg
# from tests.test_deep_rl.test_ac import play_ac

import tensorflow as tf

from reinforcement_learning.utils.plotter import plot_running_average_comparison
from reinforcement_learning.deep_RL.const import LIBRARY_TF, LIBRARY_KERAS, LIBRARY_TORCH, \
    OPTIMIZER_Adam, OPTIMIZER_RMSprop, OPTIMIZER_Adadelta, OPTIMIZER_Adagrad, OPTIMIZER_SGD
from reinforcement_learning.deep_RL.envs import CartPole, Pendulum, MountainCarContinuous, \
    LunarLander, LunarLanderContinuous, BipedalWalker, Breakout, SpaceInvaders


def perform_grid_search(lib_type=LIBRARY_TF, enable_models_saving=False, load_checkpoint=False):

    method_name = 'DQL'

    custom_env = CartPole()

    n_episodes = 3000

    ###########################################

    ep_batch_num_list = [1, 10]  # only for PG. 1 = REINFORCE algorithm (MC PG)

    fc1_dim_list = [64, 128, 256, 512]
    # fc1_dim_list = [256]

    fc2_dim_list = [64, 128, 256, 512]
    # fc2_dim_list = [256]

    # optimizer_list = [OPTIMIZER_SGD, OPTIMIZER_Adagrad, OPTIMIZER_Adadelta, OPTIMIZER_RMSprop, OPTIMIZER_Adam]
    optimizer_list = [OPTIMIZER_Adam]

    # alpha_list = [0.0005, 0.001, 0.002, 0.004]
    alpha_list = [0.0005]

    beta_list = [0.0005]  # only for DDPG & AC

    # gamma_list = [0.9, 0.95, 0.99, 1.0]
    gamma_list = [0.99]

    # eps_max_list = [1.0]
    eps_max_list = [1.0]

    # eps_min_list = [0.0, 0.01, 0.02, 0.04]
    eps_min_list = [0.1]

    # eps_dec_list = [0.1, 0.2, 0.3, 0.4]
    eps_dec_list = [1.8 / n_episodes]  # (1.0 - min) * 2 / n_episodes  # 0.996

    # eps_dec_type = utils.Calculator.EPS_DEC_LINEAR,

    # memory_size_list = [10000, 100000, 1000000]
    memory_size_list = [1000000]

    # memory_batch_size_list = [8, 16, 32, 64]
    memory_batch_size_list = [64]

    ###########################################

    counter = 0
    labels = []
    scores_histories_train = []
    scores_histories_test = []

    for optimizer_type in optimizer_list:
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

                                            fc_layers_dims = [fc1_dim, fc2_dim]
                                            double_dql = False
                                            tau = None
                                            perform_random_gameplay = False
                                            # gamma = gamma,
                                            # eps_min = eps_min, eps_dec = eps_dec,  # eps_max=eps_max,
                                            # memory_size = memory_size, memory_batch_size = memory_batch_size,

                                            agent, scores_history, scores_history_test = \
                                                play_dql(custom_env, n_episodes, fc_layers_dims, optimizer_type, alpha,
                                                         double_dql, tau,
                                                         lib_type, enable_models_saving, load_checkpoint,
                                                         perform_random_gameplay, plot=False)

                                            labels.append('[%d,%d]' % (fc1_dim, fc2_dim))
                                            scores_histories_train.append(scores_history)
                                            scores_histories_test.append(scores_history_test)

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
