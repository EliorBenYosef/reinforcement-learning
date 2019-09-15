import os

from IPython.display import clear_output
import time
import pickle

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import torch as T


LIBRARY_TF = 0
LIBRARY_TORCH = 1
LIBRARY_KERAS = 2


OPTIMIZER_Adam = 0
OPTIMIZER_RMSprop = 1
OPTIMIZER_Adadelta = 2
OPTIMIZER_Adagrad = 3
OPTIMIZER_SGD = 4


# colors = ['r--', 'g--', 'b--', 'c--', 'm--', 'y--', 'k--', 'w--']

# colors = ['#FF0000', '#fa3c3c', '#E53729',
#           '#f08228', '#FB9946', '#FF7F00',
#           '#e6af2d',
#           '#e6dc32', '#FFFF00',
#           '#a0e632', '#00FF00',  '#00dc00',
#           '#17A858', '#00d28c',
#           '#00c8c8', '#0DB0DD',  '#00a0ff', '#1e3cff', '#0000FF',
#           '#6e00dc', '#8B00FF',  '#4B0082', '#a000c8', '#662371',
#           '#f00082']

colors = ['#FF0000', '#E53729',
          '#f08228', '#FF7F00',
          '#e6af2d',
          '#e6dc32', '#FFFF00',
          '#a0e632', '#00dc00',
          '#17A858', '#00d28c',
          '#00c8c8', '#1e3cff',
          '#6e00dc', '#a000c8',
          '#f00082']


##############################################

def get_running_avg(scores, window):
    episodes = len(scores)

    x = [i + 1 for i in range(episodes)]

    running_avg = np.empty(episodes)
    for t in range(episodes):
        running_avg[t] = np.mean(scores[max(0, t - window):(t + 1)])

    return x, running_avg


def plot_running_average(env_name, method_name, scores, window=100, show=False, file_name=None):
    plt.title(env_name + ' - ' + method_name + (' - Score' if window == 0 else ' - Running Score Avg. (%d)' % window))
    plt.ylabel('Score')
    plt.xlabel('Episode')
    plt.plot(*get_running_avg(scores, window))
    if file_name:
        plt.savefig(file_name + '.png')
    if show:
        plt.show()
    plt.close()


def plot_accumulated_scores(env_name, method_name, scores, show=False, file_name=None):
    plt.title(env_name + ' - ' + method_name + ' - Accumulated Score')
    plt.ylabel('Accumulated Score')
    plt.xlabel('Episode')
    x = [i + 1 for i in range(len(scores))]
    plt.plot(x, scores)
    if file_name:
        plt.savefig(file_name + '.png')
    if show:
        plt.show()
    plt.close()


def plot_running_average_comparison(main_title, scores_list, labels=None, window=100, show=False, file_name=None):
    plt.figure(figsize=(8.5, 4.5))
    plt.title(main_title + (' - Score' if window == 0 else ' - Running Score Avg. (%d)' % window))
    plt.ylabel('Score')
    plt.xlabel('Episode')
    # colors = []
    # for i in range(len(scores_list)):
    #     colors.append(np.random.rand(3, ))
    for i, scores in enumerate(scores_list):
        plt.plot(*get_running_avg(scores, window), colors[i])
    if labels:
        # plt.legend(labels)
        plt.legend(labels, bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)
        plt.subplots_adjust(right=0.7)
    if file_name:
        plt.savefig(file_name + '.png')
    if show:
        plt.show()
    plt.close()


def plot_accumulated_scores_comparison(main_title, scores_list, labels=None, show=False, file_name=None):
    plt.figure(figsize=(8.5, 4.5))
    plt.title(main_title + ' - Accumulated Score')
    plt.ylabel('Accumulated Score')
    plt.xlabel('Episode')
    # colors = []
    # for i in range(len(scores_list)):
    #     colors.append(np.random.rand(3, ))
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
    plt.close()


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


def print_training_progress(i, ep_score, scores_history, avg_num, trailing=True, eps=None):
    print('episode: %d ;' % (i + 1), 'score: %d' % ep_score)  # score: %.2f

    eps_string = ''
    if eps:
        eps_string = 'epsilon %.3f' % eps  # %.4f

    if trailing and (i + 1) >= avg_num:
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

    print('')


##############################################

def print_v(V):
    print('\n', 'V table', '\n')
    for s in V:
        print(s, '%.5f' % V[s])


def print_q(Q):
    print('\n', 'Q table', '\n')
    # print(Q)
    for s, a in Q:
        print('s', s, 'a', a, ' - ', '%.3f' % Q[s, a])


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


def decrement_eps(eps_current, eps_min, eps_dec, eps_dec_type, eps_max=None, t=None):
    if eps_dec_type == EPS_DEC_EXPONENTIAL:
        eps_temp = eps_current * eps_dec  # eps_dec = 0.996
    elif eps_dec_type == EPS_DEC_EXPONENTIAL_TIME_RELATED and eps_max is not None and t is not None:
        return eps_min + (eps_max - eps_min) * np.exp(-eps_dec * t)  # t == i
    else:  # eps_dec_type == EPS_DEC_LINEAR:
        eps_temp = eps_current - eps_dec

    return max(eps_temp, eps_min)


##############################################

def get_plot_file_name(env_file_name, agent, beta=None, eps=False, replay_buffer=False):
    # options:
    #   .replace('.', 'p')
    #   .split('.')[1]

    gamma = 'GAMMA-' + str(agent.GAMMA).replace('.', 'p') + '_'

    fc_layers_dims = 'FC-'
    for i, fc_layer_dims in enumerate(agent.fc_layers_dims):
        if i:
            fc_layers_dims += 'x'
        fc_layers_dims += str(fc_layer_dims)
    fc_layers_dims += '_'

    if agent.optimizer_type == OPTIMIZER_Adam:
        optimizer = 'adam_'
    elif agent.optimizer_type == OPTIMIZER_RMSprop:
        optimizer = 'rmsprop_'
    elif agent.optimizer_type == OPTIMIZER_Adadelta:
        optimizer = 'adadelta_'
    elif agent.optimizer_type == OPTIMIZER_Adagrad:
        optimizer = 'adagrad_'
    else:  # agent.optimizer_type == OPTIMIZER_SGD
        optimizer = 'sgd_'
    alpha = 'alpha-' + str(agent.ALPHA).replace('.', 'p') + '_'
    beta = ('beta-' + str(beta).replace('.', 'p') + '_') if beta is not None else ''

    if eps:
        eps_max = 'max-' + str(agent.eps_max).replace('.', 'p') + '_'
        eps_min = 'min-' + str(agent.eps_min).replace('.', 'p') + '_'
        eps_dec = 'dec-' + str(agent.eps_dec).replace('.', 'p') + '_'

    if replay_buffer:
        memory_size = 'size-' + str(agent.memory_size)
        memory_batch_size = 'batch-' + str(agent.memory_batch_size)

    plot_file_name = env_file_name + '_' + gamma + fc_layers_dims + 'OPT_' + optimizer + alpha + beta
    if eps:
        plot_file_name += 'EPS_' + eps_max + eps_min + eps_dec
    if replay_buffer:
        plot_file_name += 'MEM_' + memory_size + memory_batch_size

    return plot_file_name


##############################################

def tf_set_device():
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def get_torch_device_according_to_device_type(device_type):
    # enabling GPU vs CPU:
    if device_type == 'cpu':
        device = T.device('cpu')  # default CPU. cpu:0 ?
    elif device_type == 'cuda:1':
        device = T.device('cuda:1' if T.cuda.is_available() else 'cuda')  # 2nd\default GPU. cuda:0 ?
    else:
        device = T.device('cuda' if T.cuda.is_available() else 'cpu')  # default GPU \ default CPU. :0 ?
    return device


def get_tf_session_according_to_device_type(device_type):
    if device_type is not None:
        config = tf.ConfigProto(device_count=device_type)  # {'GPU': 1}
        sess = tf.Session(config=config)
    else:
        sess = tf.Session()
    return sess


##############################################

def calc_conv_layer_output_dim(Dimension, Filter, Padding, Stride):
    return (Dimension - Filter + 2 * Padding) / Stride + 1


def calc_conv_layer_output_dims(Height, Width, Filter, Padding, Stride):
    h = (Height - Filter + 2 * Padding) // Stride + 1
    w = (Width - Filter + 2 * Padding) // Stride + 1
    return h, w


##############################################

def scale_and_normalize(np_array):
    mean = np.mean(np_array)
    std = np.std(np_array)
    if std == 0:
        std = 1
    return (np_array - mean) / std


##############################################

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
    memory_G = scale_and_normalize(memory_G)
    return memory_G


##############################################

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

def get_max_action_from_q_table(Q, s, action_space_size):
    values = np.array([Q[s, a] for a in range(action_space_size)])
    # values == Q[s, :]                                             # if Q is a numpy.ndarray
    a_max = np.random.choice(np.where(values == values.max())[0])
    return a_max


def get_policy_from_q_table(states, Q, action_space_size):
    policy = {}
    for s in states:
        policy[s] = get_max_action_from_q_table(Q, s, action_space_size)

    return policy


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
            a = get_max_action_from_q_table(Q, s, action_space_size)
            observation_, reward, done, info = env.step(a)
            eval = custom_env_object.update_evaluation_tuple(i + 1, reward, done, eval)
            ep_steps += 1
            ep_score += reward
            accumulated_score += reward
            s_ = custom_env_object.get_state(observation_)
            observation, s = observation_, s_
        total_scores[i] = ep_score
        total_accumulated_scores[i] = accumulated_score
    custom_env_object.analyze_evaluation_tuple(eval, episodes)
    return total_scores, total_accumulated_scores


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
            eval = custom_env_object.update_evaluation_tuple(i + 1, reward, done, eval)
            ep_steps += 1
            ep_score += reward
            accumulated_score += reward
            s_ = custom_env_object.get_state(observation_)
            observation, s = observation_, s_
        total_scores[i] = ep_score
        total_accumulated_scores[i] = accumulated_score
    custom_env_object.analyze_evaluation_tuple(eval, episodes)
    return total_scores, total_accumulated_scores


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
            a = get_max_action_from_q_table(Q, s, action_space_size)
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


##############################################

def pickle_load(file_name, directory=''):
    with open(directory + file_name + '.pkl', 'rb') as file:  # .pickle  # rb = read binary
        var = pickle.load(file)  # var == [X_train, y_train]
    return var


def pickle_save(var, file_name, directory=''):
    with open(directory + file_name + '.pkl', 'wb') as file:  # .pickle  # wb = write binary
        pickle.dump(var, file)  # var == [X_train, y_train]


##############################################

def query_env(env):

    print(
        'Environment Id -', env.spec.id, '\n',  # id (str): The official environment ID
        'Non-Deterministic -', env.spec.nondeterministic, '\n',  # nondeterministic (bool): Whether this environment is non-deterministic even after seeding
        'Observation Space -', env.observation_space, '\n',
        'Action Space -', env.action_space, '\n',

        'Max Episode Seconds -', env.spec.max_episode_seconds, '\n',
        'Max Episode Steps -', env.spec.max_episode_steps, '\n',  # max_episode_steps (Optional[int]): The maximum number of steps that an episode can consist of

        'Reward Range -', env.reward_range, '\n',
        'Reward Threshold -', env.spec.reward_threshold, '\n',  # reward_threshold (Optional[int]): The reward threshold before the task is considered solved

        'TimeStep Limit -', env.spec.timestep_limit, '\n',
        'Trials -', env.spec.trials, '\n',

        'Local Only -', getattr(env.spec, '_local_only', 'not defined'), '\n',
        'kwargs -', getattr(env.spec, '_kwargs', '')  # kwargs (dict): The kwargs to pass to the environment class
    )
