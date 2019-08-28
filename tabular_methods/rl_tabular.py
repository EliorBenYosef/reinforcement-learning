from gym import wrappers
from IPython.display import clear_output
import matplotlib.pyplot as plt
import numpy as np
import pickle
import time

from utils import Utils
from tabular_methods.envs_dss import Envs_DSS


class TabularMethods:

    @staticmethod
    def pickle_in(pickle_file_name):
        pickle_in = open(pickle_file_name + '.pkl', 'rb')
        Q = pickle.load(pickle_in)
        return Q

    @staticmethod
    def pickle_out(Q, pickle_file_name):
        f = open(pickle_file_name + '.pkl', "wb")
        pickle.dump(Q, f)
        f.close()

    @staticmethod
    def init_v(states):
        V = {}
        for s in states:
            V[s] = 0

        return V

    @staticmethod
    def init_q(states, action_space_size, pickle_file_name, pickle):
        if pickle:
            Q = TabularMethods.pickle_in(pickle_file_name)

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

    @staticmethod
    def init_q1_q2(states, action_space_size):
        Q1, Q2 = {}, {}
        for s in states:
            for a in range(action_space_size):
                Q1[s, a] = 0
                Q2[s, a] = 0

        return Q1, Q2

    @staticmethod
    def max_action_q(Q, s, action_space_size):
        values = np.array([Q[s, a] for a in range(action_space_size)])
        # values == Q[s, :]                                             # if Q is a numpy.ndarray

        # a_max = np.argmax(values)                                     # returns the first index with the max value
        a_max = np.random.choice(np.where(values == values.max())[0])   # returns any index with the max value

        return a_max

    @staticmethod
    def max_action_q1_q2(Q1, Q2, s, action_space_size):
        values = np.array([Q1[s, a] + Q2[s, a] for a in range(action_space_size)])

        # a_max = np.argmax(values)                                     # returns the first index with the max value
        a_max = np.random.choice(np.where(values == values.max())[0])   # returns any index with the max value

        return a_max

    @staticmethod
    def get_policy(states, Q, action_space_size):
        policy = {}
        for s in states:
            policy[s] = TabularMethods.max_action_q(Q, s, action_space_size)

        return policy

    @staticmethod
    def eps_greedy_q(Q, s, action_space_size, EPS, env):
        rand = np.random.random()
        a = TabularMethods.max_action_q(Q, s, action_space_size) \
            if rand >= EPS \
            else env.action_space.sample()
        return a

    @staticmethod
    def eps_greedy_q1_q2(Q1, Q2, s, action_space_size, EPS, env):
        rand = np.random.random()
        a = TabularMethods.max_action_q1_q2(Q1, Q2, s, action_space_size) \
            if rand >= EPS \
            else env.action_space.sample()
        return a

    @staticmethod
    def test_q(env, custom_env_object, Q, action_space_size, episodes=1000):
        total_rewards = np.zeros(episodes)
        for i in range(episodes):
            done = False
            ep_steps = 0
            ep_rewards = 0
            observation = env.reset()
            s = custom_env_object.get_state(observation)
            while not done:
                a = TabularMethods.max_action_q(Q, s, action_space_size)
                observation_, reward, done, info = env.step(a)
                ep_steps += 1
                ep_rewards += reward
                s_ = custom_env_object.get_state(observation_)
                observation, s = observation_, s_
            total_rewards[i] = ep_rewards
        plt.plot(total_rewards)
        plt.show()

    @staticmethod
    def watch_trained_agent_play(env, custom_env_object, Q, action_space_size, episodes=3, is_toy_text=False):
        # playing the best action from each state according to the Q-table

        for i in range(episodes):
            done = False
            ep_steps = 0
            ep_rewards = 0
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
                a = TabularMethods.max_action_q(Q, s, action_space_size)
                observation_, reward, done, info = env.step(a)
                ep_steps += 1
                ep_rewards += reward
                s_ = custom_env_object.get_state(observation_)
                observation, s = observation_, s_

                if is_toy_text:
                    clear_output(wait=True)
                env.render()
                if is_toy_text:
                    time.sleep(0.3)

            print('Episode reward:', ep_rewards)
            if is_toy_text:
                time.sleep(3)
                clear_output(wait=True)

        env.close()

    @staticmethod
    def calculate_episode_states_actions_returns(memory, GAMMA):
        ep_states_actions_returns = []

        G = 0
        for s, a, reward in reversed(memory):   # from end to start
            G = GAMMA * G + reward              # calculate discounted return
            ep_states_actions_returns.append((s, a, G))

        ep_states_actions_returns.reverse()
        return ep_states_actions_returns

    @staticmethod
    def calculate_episode_states_returns_by_states(memory, GAMMA):
        # another option: sort by states (each s with its G list), to calculate mean later

        ep_states_returns = {}

        G = 0
        for s, a, reward in reversed(memory):   # from end to start
            G = GAMMA * G + reward              # calculate discounted return
            if s not in ep_states_returns:
                ep_states_returns[s] = [G]
            else:
                ep_states_returns[s].append(G)

        # for g_list in ep_states_returns:
        #     g_list.reverse()

        return ep_states_returns

    class MonteCarloModel:

        def __init__(self, custom_env_object, alpha=0.1, gamma=None, episodes=50000,
                     eps_max=1.0, eps_min=None, eps_dec=0.001, eps_dec_type=Utils.EPS_DEC_LINEAR):

            self.custom_env_object = custom_env_object
            self.env = custom_env_object.env
            self.action_space_size = self.env.action_space.n
            self.states = custom_env_object.states

            self.ALPHA = alpha

            if gamma:
                self.GAMMA = gamma
            elif custom_env_object.GAMMA is not None:
                self.GAMMA = custom_env_object.GAMMA
            else:
                self.GAMMA = 0.9

            self.EPS = eps_max
            self.eps_max = eps_max
            if eps_min:
                self.eps_min = eps_min
            elif custom_env_object.EPS_MIN is not None:
                self.eps_min = custom_env_object.EPS_MIN
            else:
                self.eps_min = 0.0
            self.eps_dec = eps_dec
            self.eps_dec_type = eps_dec_type

            self.episodes = episodes
            self.totalSteps = np.zeros(episodes)
            self.totalRewards = np.zeros(episodes)
            self.accumulatedRewards = np.zeros(episodes)

            self.states_returns = {}

        def update_states_returns(self, s, G):
            if s not in self.states_returns:
                self.states_returns[s] = [G]
            else:
                self.states_returns[s].append(G)

        def calculate_episode_states_returns(self, memory, first_visit=True):
            states_visited = []

            G = 0
            for s, a, reward in reversed(memory):
                G = self.GAMMA * G + reward     # calculate discounted return

                if first_visit:
                    if s not in states_visited:
                        states_visited.append(s)
                        self.update_states_returns(s, G)
                        # V[s] += self.ALPHA / dt * (G - V[s])  # option 2
                else:  # every visit
                    self.update_states_returns(s, G)

        def perform_MC_policy_evaluation(self, policy, print_info=False, visualize=False, record=False):
            if record:
                self.env = wrappers.Monitor(
                    self.env, 'recordings/MC-PE/', force=True,
                    video_callable=lambda episode_id: episode_id == 0 or episode_id == (self.episodes - 1)
                )

            V = TabularMethods.init_v(self.states)

            dt = 1.0

            accumulated_rewards = 0

            print('\n', 'Game Started', '\n')

            for i in range(self.episodes):
                done = False
                ep_steps = 0
                ep_rewards = 0

                memory = []
                observation = self.env.reset()

                s = self.custom_env_object.get_state(observation)

                if visualize and i == self.episodes - 1:
                    self.env.render()

                while not done:
                    a = policy(s)

                    # print(observation, s, a)  # for debugging purposes

                    observation_, reward, done, info = self.env.step(a)
                    ep_steps += 1
                    ep_rewards += reward
                    accumulated_rewards += reward

                    s_ = self.custom_env_object.get_state(observation_)

                    memory.append((s, a, reward))

                    observation, s = observation_, s_

                    if visualize and i == self.episodes - 1:
                        self.env.render()

                if (i + 1) % (self.episodes // 10) == 0:
                    print('episode %d - rewards: %d, steps: %d' % (i + 1, ep_rewards, ep_steps))

                self.totalSteps[i] = ep_steps
                self.totalRewards[i] = ep_rewards
                self.accumulatedRewards[i] = accumulated_rewards

                if visualize and i == self.episodes - 1:
                    self.env.close()

                self.calculate_episode_states_returns(memory)

                # option 2
                if i % 1000 == 0:
                    dt += 0.1

            # option 1: calculate averages for each state
            for s in self.states_returns:
                V[s] = np.mean(self.states_returns[s])

            if print_info:
                Utils.print_v(V)

            print('\n', 'Game Ended', '\n')

            return V, self.totalRewards, self.accumulatedRewards

        def perform_MC_non_exploring_starts_control(self, print_info=False, visualize=False, record=False):
            # Monte Carlo control without exploring starts
            #   we use epsilon greedy with a decaying epsilon

            if record:
                self.env = wrappers.Monitor(
                    self.env, 'recordings/MC-NES/', force=True,
                    video_callable=lambda episode_id: episode_id == 0 or episode_id == (self.episodes - 1)
                )

            Q, states_actions_visited_counter = {}, {}
            for s in self.states:
                for a in range(self.action_space_size):
                    Q[s, a] = 0
                    states_actions_visited_counter[s, a] = 0

            accumulated_rewards = 0

            print('\n', 'Game Started', '\n')

            for i in range(self.episodes):
                done = False
                ep_steps = 0
                ep_rewards = 0

                memory = []
                observation = self.env.reset()

                s = self.custom_env_object.get_state(observation)

                if visualize and i == self.episodes - 1:
                    self.env.render()

                while not done:
                    a = TabularMethods.eps_greedy_q(Q, s, self.action_space_size, self.EPS, self.env)

                    observation_, reward, done, info = self.env.step(a)
                    ep_steps += 1
                    ep_rewards += reward
                    accumulated_rewards += reward

                    s_ = self.custom_env_object.get_state(observation_)

                    if isinstance(self.custom_env_object, Envs_DSS.FrozenLake):
                        a = Envs_DSS.FrozenLake.get_action(s, s_)

                    memory.append((s, a, reward))

                    observation, s = observation_, s_

                    if visualize and i == self.episodes - 1:
                        self.env.render()

                if (i + 1) % (self.episodes // 10) == 0:
                    print('episode %d - rewards: %d, steps: %d' % (i + 1, ep_rewards, ep_steps))

                self.EPS = Utils.decrement_eps(self.EPS, self.eps_min, self.eps_dec, self.eps_dec_type)

                self.totalSteps[i] = ep_steps
                self.totalRewards[i] = ep_rewards
                self.accumulatedRewards[i] = accumulated_rewards

                if visualize and i == self.episodes - 1:
                    self.env.close()

                ####################

                ep_states_actions_returns = TabularMethods.calculate_episode_states_actions_returns(memory, self.GAMMA)

                ep_states_actions_visited = []
                for s, a, G in ep_states_actions_returns:
                    if (s, a) not in ep_states_actions_visited:  # first visit
                        if not isinstance(self.custom_env_object, Envs_DSS.FrozenLake) or a != -1:
                            ep_states_actions_visited.append((s, a))
                            states_actions_visited_counter[s, a] += 1

                            # Incremental Implementation (of the update rule for the agent's estimate of the discounted future rewards)
                            #   this is a shortcut that saves you from calculating the average of a function every single time
                            #   (computationally expensive and doesn't really get you anything in terms of accuracy)
                            # new estimate = old estimate + [sample - old estimate] / N
                            Q[s, a] += (G - Q[s, a]) / states_actions_visited_counter[s, a]

            if print_info:
                Utils.print_q(Q)
                policy = TabularMethods.get_policy(self.states, Q, self.action_space_size)
                Utils.print_policy(Q, policy)

            print('\n', 'Game Ended', '\n')

            return self.totalRewards, self.accumulatedRewards

        def perform_off_policy_MC_control(self, print_info=False, visualize=False, record=False):
            # off-policy methods are the alternative to non-exploring-starts

            if record:
                self.env = wrappers.Monitor(
                    self.env, 'recordings/OP-MC/', force=True,
                    video_callable=lambda episode_id: episode_id == 0 or episode_id == (self.episodes - 1)
                )

            target_policy, Q, C = {}, {}, {}
            for s in self.states:
                target_policy[s] = self.env.action_space.sample()
                for a in range(self.action_space_size):
                    Q[s, a] = 0
                    C[s, a] = 0

            accumulated_rewards = 0

            print('\n', 'Game Started', '\n')

            for i in range(self.episodes):
                done = False
                ep_steps = 0
                ep_rewards = 0

                behavior_policy = {}
                for s in self.states:
                    rand = np.random.random()
                    behavior_policy[s] = [target_policy[s]] \
                        if rand > self.EPS \
                        else [a for a in range(self.action_space_size)]

                memory = []
                observation = self.env.reset()

                s = self.custom_env_object.get_state(observation)

                if visualize and i == self.episodes - 1:
                    self.env.render()

                while not done:
                    a = np.random.choice(behavior_policy[s])

                    observation_, reward, done, info = self.env.step(a)
                    ep_steps += 1
                    ep_rewards += reward
                    accumulated_rewards += reward

                    s_ = self.custom_env_object.get_state(observation_)

                    if isinstance(self.custom_env_object, Envs_DSS.FrozenLake):
                        a = Envs_DSS.FrozenLake.get_action(s, s_)

                    memory.append((s, a, reward))

                    observation, s = observation_, s_

                    if visualize and i == self.episodes - 1:
                        self.env.render()

                if (i + 1) % (self.episodes // 10) == 0:
                    print('episode %d - rewards: %d, steps: %d' % (i + 1, ep_rewards, ep_steps))

                self.totalSteps[i] = ep_steps
                self.totalRewards[i] = ep_rewards
                self.accumulatedRewards[i] = accumulated_rewards

                if visualize and i == self.episodes - 1:
                    self.env.close()

                ####################

                G = 0
                W = 1
                for s, a, reward in reversed(memory):   # from end to start
                    G = self.GAMMA * G + reward         # calculate discounted return

                    C[s, a] += W
                    Q[s, a] += (W / C[s, a]) * (G - Q[s, a])

                    target_policy[s] = TabularMethods.max_action_q(Q, s, self.action_space_size)

                    # taking a sub-optimal action breaks the learning loop
                    #   it only learns from greedy actions - this is a shortcoming of the class of algorithms
                    #   this makes the off-policy MC a sub-optimal strategy for MC methods
                    if a != target_policy[s]:
                        break

                    if len(behavior_policy[s]) == 1:                # agent took a greedy action
                        prob = 1 - self.EPS                         # probability of taking a greedy action.
                    else:                                           # agent took a random action
                        prob = self.EPS / len(behavior_policy[s])   # probability of taking a random action.
                    W *= 1 / prob                                   # updating the weight

                self.EPS = Utils.decrement_eps(self.EPS, self.eps_min, self.eps_dec, self.eps_dec_type)

            if print_info:
                Utils.print_q(Q)
                Utils.print_policy(Q, target_policy)

            print('\n', 'Game Ended', '\n')

            return self.totalRewards, self.accumulatedRewards

    class TdZeroModel:

        def __init__(self, custom_env_object, alpha=0.1, gamma=None, episodes=50000):

            self.custom_env_object = custom_env_object
            self.env = custom_env_object.env
            self.action_space_size = self.env.action_space.n
            self.states = custom_env_object.states

            self.ALPHA = alpha

            if gamma:
                self.GAMMA = gamma
            elif custom_env_object.GAMMA is not None:
                self.GAMMA = custom_env_object.GAMMA
            else:
                self.GAMMA = 0.9

            self.episodes = episodes
            self.totalSteps = np.zeros(episodes)
            self.totalRewards = np.zeros(episodes)
            self.accumulatedRewards = np.zeros(episodes)

        def perform_td0_policy_evaluation(self, policy, print_info=False, visualize=False, record=False):
            if record:
                self.env = wrappers.Monitor(
                    self.env, 'recordings/TD0-PE/', force=True,
                    video_callable=lambda episode_id: episode_id == 0 or episode_id == (self.episodes - 1)
                )

            V = TabularMethods.init_v(self.states)

            accumulated_rewards = 0

            print('\n', 'Game Started', '\n')

            for i in range(self.episodes):
                done = False
                ep_steps = 0
                ep_rewards = 0

                observation = self.env.reset()

                s = self.custom_env_object.get_state(observation)

                if visualize and i == self.episodes - 1:
                    self.env.render()

                while not done:
                    a = policy(s)

                    # print(observation, s, a)  # for debugging purposes

                    observation_, reward, done, info = self.env.step(a)
                    ep_steps += 1
                    ep_rewards += reward
                    accumulated_rewards += reward

                    s_ = self.custom_env_object.get_state(observation_)
                    V[s] += self.ALPHA * (reward + self.GAMMA * V[s_] - V[s])

                    # option: instead of the (V[s] += ...) line:
                    # value = weights.dot(s)
                    # value_ = weights.dot(s_)
                    # weights += self.ALPHA / dt * (reward + self.GAMMA * value_ - value) * s

                    observation, s = observation_, s_

                    if visualize and i == self.episodes - 1:
                        self.env.render()

                if (i + 1) % (self.episodes // 10) == 0:
                    print('episode %d - rewards: %d, steps: %d' % (i + 1, ep_rewards, ep_steps))

                self.totalSteps[i] = ep_steps
                self.totalRewards[i] = ep_rewards
                self.accumulatedRewards[i] = accumulated_rewards

                if visualize and i == self.episodes - 1:
                    self.env.close()

            if print_info:
                Utils.print_v(V)

            print('\n', 'Game Ended', '\n')

            return V, self.totalRewards, self.accumulatedRewards

    class GeneralModel:

        def __init__(self, custom_env_object, alpha=0.1, gamma=None, episodes=50000,
                     eps_max=1.0, eps_min=None, eps_dec=0.001, eps_dec_type=Utils.EPS_DEC_LINEAR):

            self.custom_env_object = custom_env_object
            self.env = custom_env_object.env
            self.action_space_size = self.env.action_space.n
            self.states = custom_env_object.states

            self.ALPHA = alpha

            if gamma:
                self.GAMMA = gamma
            elif custom_env_object.GAMMA is not None:
                self.GAMMA = custom_env_object.GAMMA
            else:
                self.GAMMA = 0.9

            self.EPS = eps_max
            self.eps_max = eps_max
            if eps_min:
                self.eps_min = eps_min
            elif custom_env_object.EPS_MIN is not None:
                self.eps_min = custom_env_object.EPS_MIN
            else:
                self.eps_min = 0.0
            self.eps_dec = eps_dec
            self.eps_dec_type = eps_dec_type

            self.episodes = episodes
            self.totalSteps = np.zeros(episodes)
            self.totalRewards = np.zeros(episodes)
            self.accumulatedRewards = np.zeros(episodes)

        def perform_sarsa(self, visualize=False, record=False, pickle=False):
            if record:
                self.env = wrappers.Monitor(
                    self.env, 'recordings/SARSA/', force=True,
                    video_callable=lambda episode_id: episode_id == 0 or episode_id == (self.episodes - 1)
                )

            Q = TabularMethods.init_q(self.states, self.action_space_size, self.custom_env_object.file_name, pickle)

            print('\n', 'Game Started', '\n')

            for i in range(self.episodes):
                done = False
                ep_steps = 0
                ep_rewards = 0

                observation = self.env.reset()

                s = self.custom_env_object.get_state(observation)
                a = TabularMethods.eps_greedy_q(Q, s, self.action_space_size, self.EPS, self.env)

                if visualize and i == self.episodes - 1:
                    self.env.render()

                while not done:
                    observation_, reward, done, info = self.env.step(a)
                    ep_steps += 1
                    ep_rewards += reward

                    s_ = self.custom_env_object.get_state(observation_)
                    a_ = TabularMethods.eps_greedy_q(Q, s_, self.action_space_size, self.EPS, self.env)
                    Q[s, a] += self.ALPHA * (reward + self.GAMMA * Q[s_, a_] - Q[s, a])

                    observation, s, a = observation_, s_, a_

                    if visualize and i == self.episodes - 1:
                        self.env.render()

                if (i + 1) % (self.episodes // 10) == 0:
                    print('episode %d - eps: %.2f, rewards: %d, steps: %d' % (i + 1, self.EPS, ep_rewards, ep_steps))

                self.EPS = Utils.decrement_eps(self.EPS, self.eps_min, self.eps_dec, self.eps_dec_type)

                self.totalSteps[i] = ep_steps
                self.totalRewards[i] = ep_rewards

                if visualize and i == self.episodes - 1:
                    self.env.close()

            print('\n', 'Game Ended', '\n')

            if pickle:
                TabularMethods.pickle_out(Q, self.custom_env_object.file_name)

            return self.totalRewards

        def perform_expected_sarsa(self, visualize=False, record=False, pickle=False):
            if record:
                self.env = wrappers.Monitor(
                    self.env, 'recordings/E-SARSA/', force=True,
                    video_callable=lambda episode_id: episode_id == 0 or episode_id == (self.episodes - 1)
                )

            Q = TabularMethods.init_q(self.states, self.action_space_size, self.custom_env_object.file_name, pickle)

            print('\n', 'Game Started', '\n')

            for i in range(self.episodes):
                done = False
                ep_steps = 0
                ep_rewards = 0

                observation = self.env.reset()

                s = self.custom_env_object.get_state(observation)

                if visualize and i == self.episodes - 1:
                    self.env.render()

                while not done:
                    a = TabularMethods.eps_greedy_q(Q, s, self.action_space_size, self.EPS, self.env)

                    observation_, reward, done, info = self.env.step(a)
                    ep_steps += 1
                    ep_rewards += reward

                    s_ = self.custom_env_object.get_state(observation_)
                    expected_value = np.mean(np.array([Q[s_, a] for a in range(self.action_space_size)]))
                    Q[s, a] += self.ALPHA * (reward + self.GAMMA * expected_value - Q[s, a])

                    observation, s = observation_, s_

                    if visualize and i == self.episodes - 1:
                        self.env.render()

                if (i + 1) % (self.episodes // 10) == 0:
                    print('episode %d - eps: %.2f, rewards: %d, steps: %d' % (i + 1, self.EPS, ep_rewards, ep_steps))

                self.EPS = Utils.decrement_eps(self.EPS, self.eps_min, self.eps_dec, self.eps_dec_type)

                self.totalSteps[i] = ep_steps
                self.totalRewards[i] = ep_rewards

                if visualize and i == self.episodes - 1:
                    self.env.close()

            print('\n', 'Game Ended', '\n')

            if pickle:
                TabularMethods.pickle_out(Q, self.custom_env_object.file_name)

            return self.totalRewards

        def perform_q_learning(self, visualize=False, record=False, pickle=False):
            if record:
                self.env = wrappers.Monitor(
                    self.env, 'recordings/Q-L/', force=True,
                    video_callable=lambda episode_id: episode_id == 0 or episode_id == (self.episodes - 1)
                )

            Q = TabularMethods.init_q(self.states, self.action_space_size, self.custom_env_object.file_name, pickle)

            print('\n', 'Game Started', '\n')

            for i in range(self.episodes):
                done = False
                ep_steps = 0
                ep_rewards = 0

                observation = self.env.reset()

                s = self.custom_env_object.get_state(observation)

                if visualize and i == self.episodes - 1:
                    self.env.render()

                while not done:
                    a = TabularMethods.eps_greedy_q(Q, s, self.action_space_size, self.EPS, self.env)

                    observation_, reward, done, info = self.env.step(a)
                    ep_steps += 1
                    ep_rewards += reward

                    s_ = self.custom_env_object.get_state(observation_)
                    a_ = TabularMethods.max_action_q(Q, s_, self.action_space_size)
                    Q[s, a] += self.ALPHA * (reward + self.GAMMA * Q[s_, a_] - Q[s, a])
                    # Q[s, a] += self.ALPHA * (reward + self.GAMMA * np.max(Q[s_, :]) - Q[s, a])  # if Q is a numpy.ndarray

                    observation, s = observation_, s_

                    if visualize and i == self.episodes - 1:
                        self.env.render()

                if (i + 1) % (self.episodes // 10) == 0:
                    print('episode %d - eps: %.2f, rewards: %d, steps: %d' % (i + 1, self.EPS, ep_rewards, ep_steps))

                self.EPS = Utils.decrement_eps(self.EPS, self.eps_min, self.eps_dec, self.eps_dec_type)

                self.totalSteps[i] = ep_steps
                self.totalRewards[i] = ep_rewards

                if visualize and i == self.episodes - 1:
                    self.env.close()

            print('\n', 'Game Ended', '\n')

            if pickle:
                TabularMethods.pickle_out(Q, self.custom_env_object.file_name)

            return self.totalRewards

        def perform_double_q_learning(self, visualize=False, record=False):
            if record:
                self.env = wrappers.Monitor(
                    self.env, 'recordings/D-Q-L/', force=True,
                    video_callable=lambda episode_id: episode_id == 0 or episode_id == (self.episodes - 1)
                )

            Q1, Q2 = TabularMethods.init_q1_q2(self.states, self.action_space_size)

            print('\n', 'Game Started', '\n')

            for i in range(self.episodes):
                done = False
                ep_steps = 0
                ep_rewards = 0

                observation = self.env.reset()

                s = self.custom_env_object.get_state(observation)

                if visualize and i == self.episodes - 1:
                    self.env.render()

                while not done:
                    a = TabularMethods.eps_greedy_q1_q2(Q1, Q2, s, self.action_space_size, self.EPS, self.env)

                    observation_, reward, done, info = self.env.step(a)
                    ep_steps += 1
                    ep_rewards += reward

                    s_ = self.custom_env_object.get_state(observation_)
                    rand = np.random.random()
                    if rand <= 0.5:
                        a_ = TabularMethods.max_action_q1_q2(Q1, Q1, s_, self.action_space_size)
                        Q1[s, a] += self.ALPHA * (reward + self.GAMMA * Q2[s_, a_] - Q1[s, a])
                    else:  # elif rand > 0.5
                        a_ = TabularMethods.max_action_q1_q2(Q2, Q2, s_, self.action_space_size)
                        Q2[s, a] += self.ALPHA * (reward + self.GAMMA * Q1[s_, a_] - Q2[s, a])

                    observation, s = observation_, s_

                    if visualize and i == self.episodes - 1:
                        self.env.render()

                if (i + 1) % (self.episodes // 10) == 0:
                    print('episode %d - eps: %.2f, rewards: %d, steps: %d' % (i + 1, self.EPS, ep_rewards, ep_steps))

                self.EPS = Utils.decrement_eps(self.EPS, self.eps_min, self.eps_dec, self.eps_dec_type)

                self.totalSteps[i] = ep_steps
                self.totalRewards[i] = ep_rewards

                if visualize and i == self.episodes - 1:
                    self.env.close()

            print('\n', 'Game Ended', '\n')

            return self.totalRewards

