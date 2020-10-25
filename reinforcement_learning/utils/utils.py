import os
from IPython.display import clear_output
import time
import datetime
import numpy as np
import pickle


# Printer:

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


def print_training_progress(i, ep_score, scores_history, window=100, trailing=True, eps=None, ep_start_time=None):
    time_string = ''
    if ep_start_time is not None:
        time_string = '; runtime: %s' % str(datetime.datetime.now() - ep_start_time).split('.')[0]
    print('Episode: %d ;' % (i + 1), 'score: %d' % ep_score, time_string)  # score: %.2f

    eps_string = ''
    if eps is not None:
        eps_string = 'epsilon %.3f' % eps  # %.4f

    # compute the running avg of the last 'window' / 'avg_num' episodes:
    avg_score = np.mean(scores_history[-window:]) if (i + 1) >= window else None
    if avg_score is not None:
        if trailing:  # every episode
            print('trailing %d episodes ;' % window,
                  'average score %.3f ;' % avg_score,
                  eps_string)
        elif (i + 1) % window == 0:  # every 'window' / 'avg_num' episodes
            print('episodes: %d - %d ;' % (i + 2 - window, i + 1),
                  'average score %.3f ;' % avg_score,
                  eps_string)

    return avg_score


def print_policy(Q, policy):
    print('\n', 'Policy', '\n')
    for s in policy:
        a = policy[s]
        print('s', s, 'a', a, ' - ', '%.3f' % Q[s, a])


def keras_print_model_info(keras_model):
    # model's params number
    print('Model info - total params: %d ; layers params: %s' % (
        keras_model.count_params(), [layer.count_params() for layer in keras_model.layers]
    ))

    # # model's weights and biases
    # print('model weights', '\n', model.weights, '\n')
    # print("model layers' weights and biases:", '\n', [layer.get_weights() for layer in model.layers], '\n')
    # for layer in model.layers:
    #     weights, biases = layer.get_weights()
    #     print("Layer's weights", '\n', weights, '\n')
    #     print("Layer's biases", '\n', biases, '\n')


# Calculator:

EPS_DEC_LINEAR = 0
EPS_DEC_EXPONENTIAL = 1
EPS_DEC_EXPONENTIAL_TIME_RELATED = 2
# EPS_DEC_QUADRATIC = 4


def decrement_eps(eps_current, eps_min, eps_dec, eps_dec_type, eps_max=None, t=None):
    if eps_dec_type == EPS_DEC_EXPONENTIAL:
        eps_temp = eps_current * eps_dec  # eps_dec = 0.996
    elif eps_dec_type == EPS_DEC_EXPONENTIAL_TIME_RELATED and eps_max is not None and t is not None:
        return eps_min + (eps_max - eps_min) * np.exp(-eps_dec * t)  # t == i
    else:  # eps_dec_type == Calculator.EPS_DEC_LINEAR:
        eps_temp = eps_current - eps_dec

    return max(eps_temp, eps_min)


def calculate_standardized_returns_of_consecutive_episodes(memory_r, memory_terminal, GAMMA):
    memory_G = np.zeros_like(memory_r, dtype=np.float32)  # np.float64
    G = 0
    for i in reversed(range(len(memory_r))):
        if memory_terminal[i]:
            G = 0
        G = GAMMA * G + memory_r[i]
        memory_G[i] = G
    memory_G = standardize(memory_G)
    return memory_G


# Tester:

def run_method(custom_env, episodes, choose_action):
    env = custom_env.env
    print('\n', 'Test Started', '\n')
    start_time = datetime.datetime.now()
    total_scores = np.zeros(episodes)
    total_accumulated_scores = np.zeros(episodes)
    accumulated_score = 0
    eval = custom_env.get_evaluation_tuple()
    for i in range(episodes):
        done = False
        ep_steps = 0
        ep_score = 0
        observation = env.reset()
        s = custom_env.get_state(observation)
        while not done:
            a = choose_action(s)
            observation_, reward, done, info = env.step(a)
            eval = custom_env.update_evaluation_tuple(i + 1, reward, done, eval)
            ep_steps += 1
            ep_score += reward
            accumulated_score += reward
            s_ = custom_env.get_state(observation_)
            observation, s = observation_, s_
        total_scores[i] = ep_score
        total_accumulated_scores[i] = accumulated_score
        print_training_progress(i, ep_score, total_scores)
    print('\n', 'Test Ended ~~~ Episodes: %d ~~~ Runtime: %s' %
          (episodes, str(datetime.datetime.now() - start_time).split('.')[0]), '\n')
    custom_env.analyze_evaluation_tuple(eval, episodes)
    return total_scores, total_accumulated_scores


# Watcher:

def watch_method(custom_env, episodes, choose_action, is_toy_text=False):
    env = custom_env.env

    for i in range(episodes):
        done = False
        ep_steps = 0
        ep_score = 0
        observation = env.reset()
        s = custom_env.get_state(observation)

        if is_toy_text:
            print('\n*****EPISODE ', i + 1, '*****\n')
            time.sleep(1)
            clear_output(wait=True)
        env.render()
        if is_toy_text:
            time.sleep(0.3)

        while not done:
            a = choose_action(s)
            observation_, reward, done, info = env.step(a)
            ep_steps += 1
            ep_score += reward
            s_ = custom_env.get_state(observation_)
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


# SaverLoader:

def pickle_load(file_name, directory=''):
    with open(directory + file_name + '.pkl', 'rb') as file:  # .pickle  # rb = read binary
        var = pickle.load(file)  # var == [X_train, y_train]
    return var


def pickle_save(var, file_name, directory=''):
    with open(directory + file_name + '.pkl', 'wb') as file:  # .pickle  # wb = write binary
        pickle.dump(var, file)  # var == [X_train, y_train]


# General:

def standardize(np_array):
    """
    standardize data to N(0,1)
    transforming data to have a gaussian distribution of: mean 0 (μ=0), STD 1 (σ=1)
    """
    mean = np.mean(np_array)
    std = np.std(np_array)
    if std == 0:
        std = 1
    return (np_array - mean) / std


def query_env(env):
    print(
        'Environment Id -', env.spec.id, '\n',  # id (str): The official environment ID
        # nondeterministic (bool): Whether this environment is non-deterministic even after seeding:
        'Non-Deterministic -', env.spec.nondeterministic, '\n',
        'Observation Space -', env.observation_space, '\n',
        'Action Space -', env.action_space, '\n',

        'Max Episode Seconds -', env.spec.max_episode_seconds, '\n',
        # max_episode_steps (Optional[int]): The maximum number of steps that an episode can consist of
        'Max Episode Steps -', env.spec.max_episode_steps, '\n',

        'Reward Range -', env.reward_range, '\n',
        # reward_threshold (Optional[int]): The reward threshold before the task is considered solved
        'Reward Threshold -', env.spec.reward_threshold, '\n',

        'TimeStep Limit -', env.spec.timestep_limit, '\n',
        'Trials -', env.spec.trials, '\n',

        'Local Only -', getattr(env.spec, '_local_only', 'not defined'), '\n',
        'kwargs -', getattr(env.spec, '_kwargs', '')  # kwargs (dict): The kwargs to pass to the environment class
    )


def make_sure_dir_exists(path_dir):
    if not os.path.exists(path_dir):
        os.makedirs(path_dir)
