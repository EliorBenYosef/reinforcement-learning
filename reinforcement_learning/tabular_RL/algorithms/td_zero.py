import numpy as np
from gym import wrappers

from reinforcement_learning.utils import utils
from reinforcement_learning.tabular_RL.utils import init_v, init_q, init_q1_q2, \
    max_action_q, max_action_q1_q2, eps_greedy_q, eps_greedy_q1_q2, print_v


class TD0PredictionModel:

    def __init__(self, custom_env, episodes=50000, alpha=0.1, gamma=None):

        self.custom_env = custom_env
        self.env = custom_env.envs
        self.action_space_size = self.env.action_space.n
        self.states = custom_env.states

        self.episodes = episodes
        self.totalSteps = np.zeros(episodes)
        self.totalScores = np.zeros(episodes)
        self.totalAccumulatedScores = np.zeros(episodes)

        self.ALPHA = alpha

        if gamma is not None:
            self.GAMMA = gamma
        elif custom_env.GAMMA is not None:
            self.GAMMA = custom_env.GAMMA
        else:
            self.GAMMA = 0.9

    def perform_td0_policy_evaluation(self, policy, print_info=False, visualize=False, record=False):
        if record:
            self.env = wrappers.Monitor(
                self.env, 'recordings/TD0-PE/', force=True,
                video_callable=lambda episode_id: episode_id == 0 or episode_id == (self.episodes - 1)
            )

        V = init_v(self.states)

        accumulated_scores = 0

        print('\n', 'Game Started', '\n')

        for i in range(self.episodes):
            done = False
            ep_steps = 0
            ep_score = 0

            observation = self.env.reset()

            s = self.custom_env.get_state(observation)

            if visualize and i == self.episodes - 1:
                self.env.render()

            while not done:
                a = policy(s)

                # print(observation, s, a)  # for debugging purposes

                observation_, reward, done, info = self.env.step(a)
                ep_steps += 1
                ep_score += reward
                accumulated_scores += reward

                s_ = self.custom_env.get_state(observation_)
                V[s] += self.ALPHA * (reward + self.GAMMA * V[s_] - V[s])

                # option: instead of the (V[s] += ...) line:
                # value = weights.dot(s)
                # value_ = weights.dot(s_)
                # weights += self.ALPHA / dt * (reward + self.GAMMA * value_ - value) * s

                observation, s = observation_, s_

                if visualize and i == self.episodes - 1:
                    self.env.render()

            if (i + 1) % (self.episodes // 10) == 0:
                print('episode %d - score: %d, steps: %d' % (i + 1, ep_score, ep_steps))

            self.totalSteps[i] = ep_steps
            self.totalScores[i] = ep_score
            self.totalAccumulatedScores[i] = accumulated_scores

            if visualize and i == self.episodes - 1:
                self.env.close()

        if print_info:
            print_v(V)

        print('\n', 'Game Ended', '\n')

        return V, self.totalScores, self.totalAccumulatedScores


class TD0ControlModel:
    """
    On-policy:
        SARSA
        Expected SARSA

    Off-policy:
        Q Learning
        Double Q Learning
    """

    def __init__(self, custom_env, episodes=50000, alpha=0.1, gamma=None,
                 eps_max=1.0, eps_min=None, eps_dec=None, eps_dec_type=utils.Calculator.EPS_DEC_LINEAR):

        self.custom_env = custom_env
        self.env = custom_env.envs
        self.action_space_size = self.env.action_space.n
        self.states = custom_env.states

        self.episodes = episodes
        self.totalSteps = np.zeros(episodes)
        self.totalScores = np.zeros(episodes)
        self.totalAccumulatedScores = np.zeros(episodes)

        self.ALPHA = alpha

        if gamma is not None:
            self.GAMMA = gamma
        elif custom_env.GAMMA is not None:
            self.GAMMA = custom_env.GAMMA
        else:
            self.GAMMA = 0.9

        self.EPS = eps_max
        self.eps_max = eps_max

        if eps_min is not None:
            self.eps_min = eps_min
        elif custom_env.EPS_MIN is not None:
            self.eps_min = custom_env.EPS_MIN
        else:
            self.eps_min = 0.0

        if eps_dec is not None:
            self.eps_dec = eps_dec
        else:
            # will arrive to eps_min after half the episodes:
            self.eps_dec = (self.eps_max - self.eps_min) * 2 / self.episodes

        self.eps_dec_type = eps_dec_type

    def perform_sarsa(self, visualize=False, record=False, pickle=False):
        if record:
            self.env = wrappers.Monitor(
                self.env, 'recordings/SARSA/', force=True,
                video_callable=lambda episode_id: episode_id == 0 or (i + 1) % (self.episodes // 10) == 0
            )

        Q = init_q(self.states, self.action_space_size, self.custom_env.file_name, pickle)

        accumulated_scores = 0

        print('\n', 'Game Started', '\n')

        for i in range(self.episodes):
            done = False
            ep_steps = 0
            ep_score = 0

            observation = self.env.reset()

            s = self.custom_env.get_state(observation)
            a = eps_greedy_q(Q, s, self.action_space_size, self.EPS, self.env)

            if visualize and i == self.episodes - 1:
                self.env.render()

            while not done:
                observation_, reward, done, info = self.env.step(a)
                ep_steps += 1
                ep_score += reward
                accumulated_scores += reward

                s_ = self.custom_env.get_state(observation_)
                a_ = eps_greedy_q(Q, s_, self.action_space_size, self.EPS, self.env)
                Q[s, a] += self.ALPHA * (reward + self.GAMMA * Q[s_, a_] - Q[s, a])

                observation, s, a = observation_, s_, a_

                if visualize and i == self.episodes - 1:
                    self.env.render()

            if (i + 1) % (self.episodes // 10) == 0:
                print('episode %d - eps: %.2f, score: %d, steps: %d' % (i + 1, self.EPS, ep_score, ep_steps))

            self.EPS = utils.Calculator.decrement_eps(self.EPS, self.eps_min, self.eps_dec, self.eps_dec_type)

            self.totalSteps[i] = ep_steps
            self.totalScores[i] = ep_score
            self.totalAccumulatedScores[i] = accumulated_scores

            if visualize and i == self.episodes - 1:
                self.env.close()

        print('\n', 'Game Ended', '\n')

        if pickle:
            utils.SaverLoader.pickle_save(Q, self.custom_env.file_name + '-q-table')

        return Q, self.totalScores, self.totalAccumulatedScores

    def perform_expected_sarsa(self, visualize=False, record=False, pickle=False):
        if record:
            self.env = wrappers.Monitor(
                self.env, 'recordings/E-SARSA/', force=True,
                video_callable=lambda episode_id: episode_id == 0 or episode_id == (self.episodes - 1)
            )

        Q = init_q(self.states, self.action_space_size, self.custom_env.file_name, pickle)

        accumulated_scores = 0

        print('\n', 'Game Started', '\n')

        for i in range(self.episodes):
            done = False
            ep_steps = 0
            ep_score = 0

            observation = self.env.reset()

            s = self.custom_env.get_state(observation)

            if visualize and i == self.episodes - 1:
                self.env.render()

            while not done:
                a = eps_greedy_q(Q, s, self.action_space_size, self.EPS, self.env)

                observation_, reward, done, info = self.env.step(a)
                ep_steps += 1
                ep_score += reward
                accumulated_scores += reward

                s_ = self.custom_env.get_state(observation_)
                expected_value = np.mean(np.array([Q[s_, a] for a in range(self.action_space_size)]))
                Q[s, a] += self.ALPHA * (reward + self.GAMMA * expected_value - Q[s, a])

                observation, s = observation_, s_

                if visualize and i == self.episodes - 1:
                    self.env.render()

            if (i + 1) % (self.episodes // 10) == 0:
                print('episode %d - eps: %.2f, score: %d, steps: %d' % (i + 1, self.EPS, ep_score, ep_steps))

            self.EPS = utils.Calculator.decrement_eps(self.EPS, self.eps_min, self.eps_dec, self.eps_dec_type)

            self.totalSteps[i] = ep_steps
            self.totalScores[i] = ep_score
            self.totalAccumulatedScores[i] = accumulated_scores

            if visualize and i == self.episodes - 1:
                self.env.close()

        print('\n', 'Game Ended', '\n')

        if pickle:
            utils.SaverLoader.pickle_save(Q, self.custom_env.file_name + '-q-table')

        return Q, self.totalScores, self.totalAccumulatedScores

    def perform_q_learning(self, visualize=False, record=False, pickle=False):
        if record:
            self.env = wrappers.Monitor(
                self.env, 'recordings/Q-L/', force=True,
                video_callable=lambda episode_id: episode_id == 0 or episode_id == (self.episodes - 1)
            )

        Q = init_q(self.states, self.action_space_size, self.custom_env.file_name, pickle)

        accumulated_scores = 0

        print('\n', 'Game Started', '\n')

        for i in range(self.episodes):
            done = False
            ep_steps = 0
            ep_score = 0

            observation = self.env.reset()

            s = self.custom_env.get_state(observation)

            if visualize and i == self.episodes - 1:
                self.env.render()

            while not done:
                a = eps_greedy_q(Q, s, self.action_space_size, self.EPS, self.env)

                observation_, reward, done, info = self.env.step(a)
                ep_steps += 1
                ep_score += reward
                accumulated_scores += reward

                s_ = self.custom_env.get_state(observation_)
                a_ = max_action_q(Q, s_, self.action_space_size)
                Q[s, a] += self.ALPHA * (reward + self.GAMMA * Q[s_, a_] - Q[s, a])
                # Q[s, a] += self.ALPHA * (reward + self.GAMMA * np.max(Q[s_, :]) - Q[s, a])  # if Q is a numpy.ndarray

                observation, s = observation_, s_

                if visualize and i == self.episodes - 1:
                    self.env.render()

            if (i + 1) % (self.episodes // 10) == 0:
                print('episode %d - eps: %.2f, score: %d, steps: %d' % (i + 1, self.EPS, ep_score, ep_steps))

            self.EPS = utils.Calculator.decrement_eps(self.EPS, self.eps_min, self.eps_dec, self.eps_dec_type)

            self.totalSteps[i] = ep_steps
            self.totalScores[i] = ep_score
            self.totalAccumulatedScores[i] = accumulated_scores

            if visualize and i == self.episodes - 1:
                self.env.close()

        print('\n', 'Game Ended', '\n')

        if pickle:
            utils.SaverLoader.pickle_save(Q, self.custom_env.file_name + '-q-table')

        return Q, self.totalScores, self.totalAccumulatedScores

    def perform_double_q_learning(self, visualize=False, record=False):
        if record:
            self.env = wrappers.Monitor(
                self.env, 'recordings/D-Q-L/', force=True,
                video_callable=lambda episode_id: episode_id == 0 or episode_id == (self.episodes - 1)
            )

        Q1, Q2 = init_q1_q2(self.states, self.action_space_size)

        accumulated_scores = 0

        print('\n', 'Game Started', '\n')

        for i in range(self.episodes):
            done = False
            ep_steps = 0
            ep_score = 0

            observation = self.env.reset()

            s = self.custom_env.get_state(observation)

            if visualize and i == self.episodes - 1:
                self.env.render()

            while not done:
                a = eps_greedy_q1_q2(Q1, Q2, s, self.action_space_size, self.EPS, self.env)

                observation_, reward, done, info = self.env.step(a)
                ep_steps += 1
                ep_score += reward
                accumulated_scores += reward

                s_ = self.custom_env.get_state(observation_)
                rand = np.random.random()
                if rand <= 0.5:
                    a_ = max_action_q1_q2(Q1, Q1, s_, self.action_space_size)
                    Q1[s, a] += self.ALPHA * (reward + self.GAMMA * Q2[s_, a_] - Q1[s, a])
                else:  # elif rand > 0.5
                    a_ = max_action_q1_q2(Q2, Q2, s_, self.action_space_size)
                    Q2[s, a] += self.ALPHA * (reward + self.GAMMA * Q1[s_, a_] - Q2[s, a])

                observation, s = observation_, s_

                if visualize and i == self.episodes - 1:
                    self.env.render()

            if (i + 1) % (self.episodes // 10) == 0:
                print('episode %d - eps: %.2f, score: %d, steps: %d' % (i + 1, self.EPS, ep_score, ep_steps))

            self.EPS = utils.Calculator.decrement_eps(self.EPS, self.eps_min, self.eps_dec, self.eps_dec_type)

            self.totalSteps[i] = ep_steps
            self.totalScores[i] = ep_score
            self.totalAccumulatedScores[i] = accumulated_scores

            if visualize and i == self.episodes - 1:
                self.env.close()

        print('\n', 'Game Ended', '\n')

        return Q1, Q2, self.totalScores, self.totalAccumulatedScores
