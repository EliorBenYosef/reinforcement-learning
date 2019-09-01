import os

from IPython.display import clear_output
import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import torch as T


LIBRARY_TF = 0
LIBRARY_TORCH = 1
LIBRARY_KERAS = 2


class Utils:

    @staticmethod
    def get_running_avg(scores, window):
        episodes = len(scores)

        x = [i + 1 for i in range(episodes)]

        running_avg = np.empty(episodes)
        for t in range(episodes):
            running_avg[t] = np.mean(scores[max(0, t - window):(t + 1)])

        return x, running_avg

    @staticmethod
    def plot_running_average(main_title, scores, window=100, show=False, file_name=None):
        plt.title(main_title + (' - Score' if window == 0 else ' - Running Score Avg. (%d)' % window))
        plt.ylabel('Score')
        plt.xlabel('Episode')
        plt.plot(*Utils.get_running_avg(scores, window))
        if file_name:
            plt.savefig(file_name + '.png')
        if show:
            plt.show()

    @staticmethod
    def plot_accumulated_scores(main_title, scores, show=False, file_name=None):
        plt.title(main_title + ' - Accumulated Score')
        plt.ylabel('Accumulated Score')
        plt.xlabel('Episode')
        x = [i + 1 for i in range(len(scores))]
        plt.plot(x, scores)
        if file_name:
            plt.savefig(file_name + '.png')
        if show:
            plt.show()

    @staticmethod
    def plot_running_average_comparison(main_title, scores_list, labels=None, window=100, show=False, file_name=None):
        plt.figure(figsize=(8.5, 4.5))
        plt.title(main_title + (' - Score' if window == 0 else ' - Running Score Avg. (%d)' % window))
        plt.ylabel('Score')
        plt.xlabel('Episode')
        colors = ['r--', 'g--', 'b--', 'c--', 'm--', 'y--', 'k--']
        for i, scores in enumerate(scores_list):
            plt.plot(*Utils.get_running_avg(scores, window), colors[i])
        if labels:
            # plt.legend(labels)
            plt.legend(labels, bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)
            plt.subplots_adjust(right=0.7)
        if file_name:
            plt.savefig(file_name + '.png')
        if show:
            plt.show()

    @staticmethod
    def plot_accumulated_scores_comparison(main_title, scores_list, labels=None, show=False, file_name=None):
        plt.figure(figsize=(8.5, 4.5))
        plt.title(main_title + ' - Accumulated Score')
        plt.ylabel('Accumulated Score')
        plt.xlabel('Episode')
        colors = ['r--', 'g--', 'b--', 'c--', 'm--', 'y--', 'k--']

        for i, scores in enumerate(scores_list):
            x = [i + 1 for i in range(len(scores))]
            plt.plot(x, scores, colors[i])
        if labels:
            # plt.legend(labels)
            plt.legend(labels, bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)
            plt.subplots_adjust(right=0.7)
        if file_name:
            plt.savefig(file_name + '.png')
        if show:
            plt.show()

    @staticmethod
    def print_average_score(total_scores, ratio=10):
        # Calculate and print the average score per a number of episodes (tick)

        scores_per_tick_episodes = np.split(np.array(total_scores), ratio)  # episodes / tick

        episodes = len(total_scores)
        tick = episodes // ratio
        print('\n********Average score per %d episodes********\n' % tick)
        count = tick
        for r in scores_per_tick_episodes:
            print(count, ": ", str(sum(r / 1000)))
            count += tick

    @staticmethod
    def print_training_progress(i, ep_score, scores_history, avg_num, trailing=True, eps=None):
        print('')
        print('episode: %d ;' % (i + 1), 'score: %d' % ep_score)  # score: %.2f

        eps_string = ''
        if eps:
            eps_string = 'epsilon %.3f' % eps  # %.4f

        if trailing and i >= avg_num:
            # gives you the running avg of the last 'avg_num' episodes, every episode:
            avg_score = np.mean(scores_history[-avg_num:])
            print('trailing %d episodes ;' % avg_num,
                  'average score %.3f ;' % avg_score,
                  eps_string)

        elif (i + 1) % avg_num == 0:
            # gives you the running avg of the last 'avg_num' episodes, every 'avg_num' episodes:
            avg_score = np.mean(scores_history[max(0, i + 1 - avg_num):(i + 1)])
            print('episodes: %d - %d ;' % (i + 2 - avg_num, i + 1),
                  'average score %.3f ;' % avg_score,
                  eps_string)

    ##############################################

    @staticmethod
    def plot_running_average_scatter(ax, total_scores, window, x=None):
        # if x is None:
        #     # x = [i for i in range(len(total_scores))]
        #     x = [i + 1 for i in range(len(total_scores))]

        # ax.scatter(x, Utils.get_running_avg(total_scores, window), color="C1")
        ax.scatter(*Utils.get_running_avg(total_scores, window), color="C1")
        # ax.xaxis.tick_top()
        ax.axes.get_xaxis().set_visible(False)
        ax.yaxis.tick_right()
        # ax.set_xlabel('x label 2', color="C1")
        ax.set_ylabel('Score', color="C1")
        # ax.xaxis.set_label_position('top')
        ax.yaxis.set_label_position('right')
        # ax.tick_params(axis='x', colors="C1")
        ax.tick_params(axis='y', colors="C1")

    @staticmethod
    def plot_eps_history(ax, eps_history, x=None):
        if x is None:
            # x = [i for i in range(len(eps_history))]
            x = [i + 1 for i in range(len(eps_history))]

        ax.plot(x, eps_history, color="C0")
        ax.set_xlabel("Episode", color="C0")
        ax.set_ylabel("Epsilon", color="C0")
        ax.tick_params(axis='x', colors="C0")
        ax.tick_params(axis='y', colors="C0")

    @staticmethod
    def plot_eps_history_and_running_avg(main_title, total_scores, eps_history, window=100, show=False, file_name=None):
        plt.title(main_title)
        fig = plt.figure()

        label = ' - Score' if window == 0 else ' - Running Score Avg. (%d)' % window
        ax01 = fig.add_subplot(111, label=label, frame_on=False)
        Utils.plot_running_average_scatter(ax01, total_scores, window)

        ax02 = fig.add_subplot(111, label='Epsilon History')
        Utils.plot_eps_history(ax02, eps_history)

        if file_name:
            plt.savefig(file_name + '.png')
        if show:
            plt.show()

    ##############################################

    @staticmethod
    def print_v(V):
        print('\n', 'V table', '\n')
        for s in V:
            print(s, '%.5f' % V[s])

    @staticmethod
    def print_q(Q):
        print('\n', 'Q table', '\n')
        # print(Q)
        for s, a in Q:
            print('s', s, 'a', a, ' - ', '%.3f' % Q[s, a])

    @staticmethod
    def print_policy(Q, policy):
        print('\n', 'Policy', '\n')
        for s in policy:
            a = policy[s]
            print('s', s, 'a', a, ' - ', '%.3f' % Q[s, a])

    ##############################################

    EPS_DEC_LINEAR = 0
    EPS_DEC_EXPONENTIAL = 1
    EPS_DEC_EXPONENTIAL_TIME_RELATED = 2
    # EPS_DEC_QUADRATIC = 4

    @staticmethod
    def decrement_eps(eps_current, eps_min, eps_dec, eps_dec_type, eps_max=None, t=None):
        if eps_dec_type == Utils.EPS_DEC_EXPONENTIAL:
            eps_temp = eps_current * eps_dec
        elif eps_dec_type == Utils.EPS_DEC_EXPONENTIAL_TIME_RELATED and eps_max is not None and t is not None:
            return eps_min + (eps_max - eps_min) * np.exp(-eps_dec * t)  # t == i
        else:  # eps_dec_type == Utils.EPS_DEC_LINEAR:
            eps_temp = eps_current - eps_dec

        return max(eps_temp, eps_min)

    ##############################################

    @staticmethod
    def get_plot_file_name(custom_env, fc_layers_dims, alpha, beta=None):
        # episodes = 'episodes-' + str(self.episodes)
        # memory_size = 'memorySize-' + str(self.memory_size)
        alpha_str = 'alpha-' + str(alpha).replace('.', 'p')  # .split('.')[1]
        beta_str = 'beta-' + str(beta).replace('.', 'p') if beta else ''  # .split('.')[1]
        gamma_str = 'gamma-' + str(custom_env.GAMMA).replace('.', 'p')  # .split('.')[1]
        fc_layers_dims_str = 'fc-'
        for i, fc_layer_dims in enumerate(fc_layers_dims):
            if i:
                fc_layers_dims_str += 'x'
            fc_layers_dims_str += str(fc_layer_dims)
        plot_file_name = custom_env.file_name + '_' + alpha_str + '_' + beta_str + '_' + gamma_str + '_' + \
                         fc_layers_dims_str
                         # + '_' + '-bs64-adam-faster_decay'
        return plot_file_name

    ##############################################

    @staticmethod
    def init_seeds(lib_type, env, env_seed, np_seed, tf_seed):
        if lib_type == LIBRARY_TF:
            # use environment variables, to specify which GPU should models run on, in tensorflow.
            # gets the graphics cards' bus ids from the motherboards, and lets the DEVICE_ORDER know what it is.
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            # puts it on first GPU, with respect to the previous order.
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"

            tf.set_random_seed(tf_seed)

        # set the numpy random number generator seed, to improve reproducibility
        #   (because this is an inherently probabilistic model).
        # not seeding it --> uses the system clock as a seed --> different random seeds every run.
        np.random.seed(np_seed)

        env.seed(env_seed)

    ##############################################

    @staticmethod
    def get_torch_device_according_to_device_type(device_type):
        # enabling GPU vs CPU:
        if device_type == 'cpu':
            device = T.device('cpu')  # default CPU. cpu:0 ?
        elif device_type == 'cuda:1':
            device = T.device('cuda:1' if T.cuda.is_available() else 'cuda')  # 2nd\default GPU. cuda:0 ?
        else:
            device = T.device('cuda' if T.cuda.is_available() else 'cpu')  # default GPU \ default CPU. :0 ?
        return device

    @staticmethod
    def get_tf_session_according_to_device_type(device_type):
        if device_type is not None:
            config = tf.ConfigProto(device_count=device_type)  # {'GPU': 1}
            sess = tf.Session(config=config)
        else:
            sess = tf.Session()
        return sess

    ##############################################

    @staticmethod
    def calc_conv_layer_output_dim(Dimension, Filter, Padding, Stride):
        return (Dimension - Filter + 2 * Padding) / Stride + 1

    @staticmethod
    def calc_conv_layer_output_dims(Height, Width, Filter, Padding, Stride):
        h = (Height - Filter + 2 * Padding) // Stride + 1
        w = (Width - Filter + 2 * Padding) // Stride + 1
        return h, w

    ##############################################

    @staticmethod
    def scale_and_normalize(np_array):
        mean = np.mean(np_array)
        std = np.std(np_array)
        if std == 0:
            std = 1
        return (np_array - mean) / std

    ##############################################

    @staticmethod
    def calculate_returns_of_consecutive_episodes(memory_r, memory_terminal, GAMMA):
        memory_G = []
        G = 0
        n = len(memory_r)
        for reward in reversed(memory_r):
            if memory_terminal[n - 1 - len(memory_G)]:
                G = 0
            G = GAMMA * G + reward
            memory_G.append(G)
        memory_G = np.flip(np.array(memory_G, dtype=np.float64), 0)
        memory_G = Utils.scale_and_normalize(memory_G)
        return memory_G

    ##############################################

    @staticmethod
    def compare_current_and_original_params(current_actor, current_critic,
                                            original_actor, original_critic):
        current_actor_dict = dict(current_actor.named_parameters())
        original_actor_dict = dict(original_actor.named_parameters())
        print('Checking Actor parameters')
        for param in current_actor_dict:
            print(param, T.equal(original_actor_dict[param], current_actor_dict[param]))

        current_critic_dict = dict(current_critic.named_parameters())
        original_critic_dict = dict(original_critic.named_parameters())
        print('Checking critic parameters')
        for param in current_critic_dict:
            print(param, T.equal(original_critic_dict[param], current_critic_dict[param]))

        input()

    ##############################################

    @staticmethod
    def get_max_action_from_q_table(Q, s, action_space_size):
        values = np.array([Q[s, a] for a in range(action_space_size)])
        # values == Q[s, :]                                             # if Q is a numpy.ndarray
        a_max = np.random.choice(np.where(values == values.max())[0])
        return a_max

    @staticmethod
    def get_policy_from_q_table(states, Q, action_space_size):
        policy = {}
        for s in states:
            policy[s] = Utils.get_max_action_from_q_table(Q, s, action_space_size)

        return policy

    @staticmethod
    def test_q_table(custom_env_object, Q, episodes=1000):
        env = custom_env_object.env
        action_space_size = env.action_space.n

        total_scores = np.zeros(episodes)
        total_accumulated_scores = np.zeros(episodes)
        accumulated_score = 0
        eval = custom_env_object.get_evaluation_tuple()
        for i in range(episodes):
            done = False
            ep_steps = 0
            ep_score = 0
            observation = env.reset()
            s = custom_env_object.get_state(observation)
            while not done:
                a = Utils.get_max_action_from_q_table(Q, s, action_space_size)
                observation_, reward, done, info = env.step(a)
                eval = custom_env_object.update_evaluation_tuple(i+1, reward, done, eval)
                ep_steps += 1
                ep_score += reward
                accumulated_score += reward
                s_ = custom_env_object.get_state(observation_)
                observation, s = observation_, s_
            total_scores[i] = ep_score
            total_accumulated_scores[i] = accumulated_score
        custom_env_object.analyze_evaluation_tuple(eval, episodes)
        return total_scores, total_accumulated_scores

    @staticmethod
    def test_policy(custom_env_object, policy, episodes=1000):
        env = custom_env_object.env

        total_scores = np.zeros(episodes)
        total_accumulated_scores = np.zeros(episodes)
        accumulated_score = 0
        eval = custom_env_object.get_evaluation_tuple()
        for i in range(episodes):
            done = False
            ep_steps = 0
            ep_score = 0
            observation = env.reset()
            s = custom_env_object.get_state(observation)
            while not done:
                a = policy[s]
                observation_, reward, done, info = env.step(a)
                eval = custom_env_object.update_evaluation_tuple(i+1, reward, done, eval)
                ep_steps += 1
                ep_score += reward
                accumulated_score += reward
                s_ = custom_env_object.get_state(observation_)
                observation, s = observation_, s_
            total_scores[i] = ep_score
            total_accumulated_scores[i] = accumulated_score
        custom_env_object.analyze_evaluation_tuple(eval, episodes)
        return total_scores, total_accumulated_scores

    @staticmethod
    def watch_trained_agent_play(custom_env_object, Q, action_space_size, episodes=3, is_toy_text=False):
        # playing the best action from each state according to the Q-table

        env = custom_env_object.env

        for i in range(episodes):
            done = False
            ep_steps = 0
            ep_score = 0
            observation = env.reset()
            s = custom_env_object.get_state(observation)

            if is_toy_text:
                print('\n*****EPISODE ', i + 1, '*****\n')
                time.sleep(1)
                clear_output(wait=True)
            env.render()
            if is_toy_text:
                time.sleep(0.3)

            while not done:
                a = Utils.get_max_action_from_q_table(Q, s, action_space_size)
                observation_, reward, done, info = env.step(a)
                ep_steps += 1
                ep_score += reward
                s_ = custom_env_object.get_state(observation_)
                observation, s = observation_, s_

                if is_toy_text:
                    clear_output(wait=True)
                env.render()
                if is_toy_text:
                    time.sleep(0.3)

            print('Episode Score:', ep_score)
            if is_toy_text:
                time.sleep(3)
                clear_output(wait=True)

        env.close()

