import numpy as np
from gym import wrappers

from reinforcement_learning.utils.utils import decrement_eps, EPS_DEC_LINEAR, print_policy
from reinforcement_learning.tabular_RL.utils import init_v, init_q1_q2, \
    max_action_q, eps_greedy_q, print_v, print_q, get_policy_table_from_q_table, \
    calculate_episode_states_actions_returns


class MCPredictionModel:

    def __init__(self, custom_env, episodes=50000, alpha=0.1, gamma=None):

        self.custom_env = custom_env
        self.env = custom_env.envs
        self.states = custom_env.states

        self.episodes = episodes
        self.totalSteps = np.zeros(episodes)
        self.totalScores = np.zeros(episodes)
        self.totalAccumulatedScores = np.zeros(episodes)
        self.states_returns = {}

        self.ALPHA = alpha

        if gamma is not None:
            self.GAMMA = gamma
        elif custom_env.GAMMA is not None:
            self.GAMMA = custom_env.GAMMA
        else:
            self.GAMMA = 0.9

    def perform_mc_policy_evaluation(self, policy, print_info=False, visualize=False, record=False):
        if record:
            self.env = wrappers.Monitor(
                self.env, 'recordings/MC-PE/', force=True,
                video_callable=lambda episode_id: episode_id == 0 or episode_id == (self.episodes - 1)
            )

        V = init_v(self.states)

        dt = 1.0

        accumulated_scores = 0

        print('\n', 'Game Started', '\n')

        for i in range(self.episodes):
            done = False
            ep_steps = 0
            ep_score = 0

            memory = []
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

                memory.append((s, a, reward))

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

            self.calculate_episode_states_returns(memory)

            # option 2
            if i % 1000 == 0:
                dt += 0.1

        # option 1: calculate averages for each state
        for s in self.states_returns:
            V[s] = np.mean(self.states_returns[s])

        if print_info:
            print_v(V)

        print('\n', 'Game Ended', '\n')

        return V, self.totalScores, self.totalAccumulatedScores

    def calculate_episode_states_returns(self, memory, first_visit=True):
        states_visited = []

        G = 0
        for s, a, reward in reversed(memory):
            G = self.GAMMA * G + reward  # calculate discounted return

            if first_visit:
                if s not in states_visited:
                    states_visited.append(s)
                    self.update_states_returns(s, G)
                    # V[s] += self.ALPHA / dt * (G - V[s])  # option 2
            else:  # every visit
                self.update_states_returns(s, G)

    def update_states_returns(self, s, G):
        if s not in self.states_returns:
            self.states_returns[s] = [G]
        else:
            self.states_returns[s].append(G)


class MCControlModel:

    def __init__(self, custom_env, episodes=50000, alpha=0.1, gamma=None,
                 eps_max=1.0, eps_min=None, eps_dec=None, eps_dec_type=EPS_DEC_LINEAR):

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

    def perform_mc_non_exploring_starts_control(self, print_info=False, visualize=False, record=False):
        # Monte Carlo control without exploring starts
        #   we use epsilon greedy with a decaying epsilon

        if record:
            self.env = wrappers.Monitor(
                self.env, 'recordings/MC-NES/', force=True,
                video_callable=lambda episode_id: episode_id == 0 or episode_id == (self.episodes - 1)
            )

        Q, states_actions_visited_counter = init_q1_q2(self.states, self.action_space_size)
        accumulated_scores = 0

        print('\n', 'Game Started', '\n')

        for i in range(self.episodes):
            done = False
            ep_steps = 0
            ep_score = 0

            memory = []
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

                memory.append((s, a, reward))

                observation, s = observation_, s_

                if visualize and i == self.episodes - 1:
                    self.env.render()

            if (i + 1) % (self.episodes // 10) == 0:
                print('episode %d - score: %d, steps: %d' % (i + 1, ep_score, ep_steps))

            self.EPS = decrement_eps(self.EPS, self.eps_min, self.eps_dec, self.eps_dec_type)

            self.totalSteps[i] = ep_steps
            self.totalScores[i] = ep_score
            self.totalAccumulatedScores[i] = accumulated_scores

            if visualize and i == self.episodes - 1:
                self.env.close()

            ####################

            ep_states_actions_returns = calculate_episode_states_actions_returns(memory, self.GAMMA)

            ep_states_actions_visited = []
            for s, a, G in ep_states_actions_returns:
                if (s, a) not in ep_states_actions_visited:  # first visit
                    ep_states_actions_visited.append((s, a))
                    states_actions_visited_counter[s, a] += 1

                    # Incremental Implementation
                    # (of the update rule for the agent's estimate of the discounted future rewards)
                    #   this is a shortcut that saves you from calculating the average of a function every single time
                    #   (computationally expensive and doesn't really get you anything in terms of accuracy)
                    # new estimate = old estimate + [sample - old estimate] / N
                    Q[s, a] += (G - Q[s, a]) / states_actions_visited_counter[s, a]

        policy = get_policy_table_from_q_table(self.states, Q, self.action_space_size)

        if print_info:
            print_q(Q)
            print_policy(Q, policy)

        print('\n', 'Game Ended', '\n')

        return policy, self.totalScores, self.totalAccumulatedScores

    def perform_off_policy_mc_control(self, print_info=False, visualize=False, record=False):
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

        accumulated_scores = 0

        print('\n', 'Game Started', '\n')

        for i in range(self.episodes):
            done = False
            ep_steps = 0
            ep_score = 0

            behavior_policy = {}
            for s in self.states:
                rand = np.random.random()
                behavior_policy[s] = [target_policy[s]] \
                    if rand > self.EPS \
                    else [a for a in range(self.action_space_size)]

            memory = []
            observation = self.env.reset()

            s = self.custom_env.get_state(observation)

            if visualize and i == self.episodes - 1:
                self.env.render()

            while not done:
                a = np.random.choice(behavior_policy[s])

                observation_, reward, done, info = self.env.step(a)
                ep_steps += 1
                ep_score += reward
                accumulated_scores += reward

                s_ = self.custom_env.get_state(observation_)

                memory.append((s, a, reward))

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

            ####################

            G = 0
            W = 1
            for s, a, reward in reversed(memory):  # from end to start
                G = self.GAMMA * G + reward  # calculate discounted return

                C[s, a] += W
                Q[s, a] += (W / C[s, a]) * (G - Q[s, a])

                target_policy[s] = max_action_q(Q, s, self.action_space_size)

                # taking a sub-optimal action breaks the learning loop
                #   it only learns from greedy actions - this is a shortcoming of the class of algorithms
                #   this makes the off-policy MC a sub-optimal strategy for MC methods
                if a != target_policy[s]:
                    break

                if len(behavior_policy[s]) == 1:  # agent took a greedy action
                    prob = 1 - self.EPS  # probability of taking a greedy action.
                else:  # agent took a random action
                    prob = self.EPS / len(behavior_policy[s])  # probability of taking a random action.
                W *= 1 / prob  # updating the weight

            self.EPS = decrement_eps(self.EPS, self.eps_min, self.eps_dec, self.eps_dec_type)

        if print_info:
            print_q(Q)
            print_policy(Q, target_policy)

        print('\n', 'Game Ended', '\n')

        return target_policy, self.totalScores, self.totalAccumulatedScores
