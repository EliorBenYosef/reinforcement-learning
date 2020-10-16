import numpy as np

from reinforcement_learning.utils.utils import SaverLoader, Tester, Watcher


# Initialization:

def init_v(states):
    V = {}
    for s in states:
        V[s] = 0
    return V


def init_q(states, action_space_size, pickle_file_name, pickle):
    if pickle:
        Q = SaverLoader.pickle_load(pickle_file_name)
    else:
        # if Q is a numpy.ndarray, options:
        #   Q = np.zeros((state_space_size, action_space_size))
        #   Q = np.ones((state_space_size, action_space_size))
        #   Q = np.random.random((state_space_size, action_space_size))
        Q = {}
        for s in states:
            for a in range(action_space_size):
                Q[s, a] = 0
    return Q


def init_q1_q2(states, action_space_size):
    Q1, Q2 = {}, {}
    for s in states:
        for a in range(action_space_size):
            Q1[s, a] = 0
            Q2[s, a] = 0
    return Q1, Q2


# Action Selection:

def max_action_q(Q, s, action_space_size):
    values = np.array([Q[s, a] for a in range(action_space_size)])
    # values == Q[s, :]                                             # if Q is a numpy.ndarray
    a_max = np.random.choice(np.where(values == values.max())[0])
    return a_max


def max_action_q1_q2(Q1, Q2, s, action_space_size):
    values = np.array([Q1[s, a] + Q2[s, a] for a in range(action_space_size)])
    a_max = np.random.choice(np.where(values == values.max())[0])
    return a_max


def eps_greedy_q(Q, s, action_space_size, EPS, env):
    rand = np.random.random()
    a = max_action_q(Q, s, action_space_size) \
        if rand >= EPS \
        else env.action_space.sample()
    return a


def eps_greedy_q1_q2(Q1, Q2, s, action_space_size, EPS, env):
    rand = np.random.random()
    a = max_action_q1_q2(Q1, Q2, s, action_space_size) \
        if rand >= EPS \
        else env.action_space.sample()
    return a


# Calculations:

def calculate_episode_states_actions_returns(memory, GAMMA):
    ep_states_actions_returns = []

    G = 0
    for s, a, reward in reversed(memory):  # from end to start
        G = GAMMA * G + reward  # calculate discounted return
        ep_states_actions_returns.append((s, a, G))

    ep_states_actions_returns.reverse()
    return ep_states_actions_returns


def calculate_episode_states_returns_by_states(memory, GAMMA):
    # another option: sort by states (each s with its G list), to calculate mean later

    ep_states_returns = {}

    G = 0
    for s, a, reward in reversed(memory):  # from end to start
        G = GAMMA * G + reward  # calculate discounted return
        if s not in ep_states_returns:
            ep_states_returns[s] = [G]
        else:
            ep_states_returns[s].append(G)

    # for g_list in ep_states_returns:
    #     g_list.reverse()

    return ep_states_returns


# Miscellaneous:

def get_policy_table_from_q_table(states, Q, action_space_size):
    policy = {}
    for s in states:
        policy[s] = max_action_q(Q, s, action_space_size)
    return policy


def test_q_table(custom_env, Q, episodes=1000):
    return Tester.test_method(custom_env, episodes, lambda s: max_action_q(Q, s, custom_env.envs.action_space.n))


def test_policy_table(custom_env, policy, episodes=1000):
    return Tester.test_method(custom_env, episodes, lambda s: policy[s])


def watch_q_table(custom_env, Q, episodes=3):
    Watcher.watch_method(custom_env, episodes, lambda s: max_action_q(Q, s, custom_env.envs.action_space.n))


# Print:

def print_v(V):
    print('\n', 'V table', '\n')
    for s in V:
        print(s, '%.5f' % V[s])


def print_q(Q):
    print('\n', 'Q table', '\n')
    # print(Q)
    for s, a in Q:
        print('s', s, 'a', a, ' - ', '%.3f' % Q[s, a])
