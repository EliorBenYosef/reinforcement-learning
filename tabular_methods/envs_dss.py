import gym
import numpy as np
import matplotlib.pyplot as plt


# https://github.com/openai/gym/wiki/Table-of-environments

# Environments with Discretized State Space

class Envs_DSS:

    # Toy Text - Grid World, Windy Gridworld

    class FrozenLake:

        # the agent walks on a 4x4 matrix.
        # there's a single starting point (S), and a single goal (G) where the frisbee is located.
        # the agent can walk on the frozen surface (F), but falls in the hole (H) to its doom.
        # there's a chance for the agent to slip and not go in the chosen direction of the action.
        # The episode ends when you reach the goal or fall in a hole.

        # Rewards:
        # You receive a reward of 1 if you reach the goal, and zero otherwise.

        # Actions (4): left (0), down (1), right (2), up (3)

        def __init__(self):
            self.name = 'Frozen Lake'
            self.file_name = 'frozen-lake-v0'
            self.env = gym.make('FrozenLake-v0')

            self.GAMMA = 0.9  # 0.99 ?
            self.EPS_MIN = 0.1

            # State space analysis
            self.rows = 4
            self.columns = 4

            # Construct state space
            self.states = []
            for r in range(self.rows):
                for c in range(self.columns):
                    self.states.append((r, c))

        def get_state(self, observation):
            c = observation % self.columns
            r = (observation // self.columns) % self.rows

            return r, c

        def test_policy(self, policy, episodes=1000):
            total_accumulated_rewards = np.zeros(episodes)
            accumulated_rewards = 0

            success = 0

            for i in range(episodes):
                done = False
                ep_rewards = 0
                observation = self.env.reset()
                s = self.get_state(observation)
                while not done:
                    a = policy[s]
                    observation_, reward, done, info = self.env.step(a)
                    s_ = self.get_state(observation_)
                    ep_rewards += reward
                    accumulated_rewards += reward
                    s, observation = s_, observation_
                total_accumulated_rewards[i] = accumulated_rewards

                if ep_rewards == 1:
                    success += 1

            print('success rate: %d%%' % (success * 100 / episodes))

            plt.plot(total_accumulated_rewards)
            plt.show()

    class Taxi:

        # The taxi drives on a 5x5 matrix.
        # your job is to pick up the passenger at one location and drop him off in another.

        # There are:
        #   4 possible pick-up\drop-off destinations (R, G, B, Y).
        #   5 possible passenger locations (4 destinations +  Taxi).
        # The pipe characters '|' indicate obstacles (the taxi cannot drive through them).

        # Rewards:
        # +20 points for a successful drop-off.
        # -1 point for every time-step it takes.
        # -10 points for illegal pick-up and drop-off actions.
        # -1 points for driving against a wall.

        # Actions (6): north (0), south (1), east (2), west (3), pick up (4), drop off (5)

        def __init__(self):
            self.name = 'Taxi'
            self.file_name = 'taxi-v2'
            self.env = gym.make('Taxi-v2')

            self.GAMMA = 0.999
            self.EPS_MIN = 0.0

            # State space analysis
            self.taxi_rows = 5
            self.taxi_columns = 5
            self.passenger_locations = 5  # 4 starting locations + the taxi
            self.destinations = 4

            # Construct state space
            # ((taxi_row*5 + taxi_col)*5 + passenger_location)*4 + destination.
            self.states = []
            for tr in range(self.taxi_rows):
                for tc in range(self.taxi_columns):
                    for pl in range(self.passenger_locations):
                        for d in range(self.destinations):
                            self.states.append((tr, tc, pl, d))

        def get_state(self, observation):
            d = observation % self.destinations
            pl = (observation // self.destinations) % self.passenger_locations
            tc = ((observation // self.destinations) // self.passenger_locations) % self.taxi_columns
            tr = (((observation // self.destinations) // self.passenger_locations) // self.taxi_columns) % self.taxi_rows

            return tr, tc, pl, d

    class Blackjack:

        # At the start, the player receives two cards (so the total min is 2 + 2 = 4)
        # Object: Have your card sum be greater than the dealers without exceeding 21.

        # Reward: –1 for losing, 0 for a draw, and >=1 for winning (1.5 for natural = getting 21 on the first deal)

        # Actions (2): stick = stop receiving cards (0), hit \ receive another card (1)

        def __init__(self):
            self.name = 'Blackjack'
            self.file_name = 'blackjack-v0'
            self.env = gym.make('Blackjack-v0')

            self.GAMMA = 1.0
            self.EPS_MIN = 0.0

            # State space analysis
            self.agentCardsSumSpace = [i for i in range(4, 22)]  # agent's cards sum - sum of the player's cards
            self.dealerShowingCardSpace = [i + 1 for i in range(10)]  # dealer's showing card - the card that the dealer has showing, 1 = Ace, 10 = face card
            self.agentUsableAceSpace = [False, True]  # agent's usable ace - if the player has usable ace, it can count as 1 \ 11

            # Construct state space
            self.states = []
            for total in self.agentCardsSumSpace:
                for card in self.dealerShowingCardSpace:
                    for ace in self.agentUsableAceSpace:
                        self.states.append((total, card, ace))

        def get_state(self, observation):
            # observation == agentCardsSum, dealerShowingCard, agentUsableAce
            return observation

        def test_policy(self, policy, episodes=1000):
            accumulated_rewards_total = np.zeros(episodes)
            accumulated_rewards = 0

            wins = 0
            losses = 0
            draws = 0

            for i in range(episodes):
                done = False
                ep_rewards = 0
                s = self.env.reset()
                while not done:
                    a = policy[s]
                    s_, reward, done, info = self.env.step(a)
                    ep_rewards += reward
                    accumulated_rewards += reward
                    s = s_
                accumulated_rewards_total[i] = accumulated_rewards

                if ep_rewards >= 1:
                    wins += 1
                elif ep_rewards == 0:
                    draws += 1
                elif ep_rewards == -1:
                    losses += 1

            print('win rate', wins / episodes,
                  'loss rate', losses / episodes,
                  'draw rate', draws / episodes)

            plt.plot(accumulated_rewards_total)
            plt.show()

    # Custom Grid World

    # Classic Control

    class CartPole:

        # AKA "Inverted Pendulum".

        # Actions (2): go left (0), go right (1)

        CART_X = 0
        CART_X_VEL = 1
        POLE_THETA = 2
        POLE_THETA_VEL = 3

        def __init__(self, single_state_space=-1, pole_theta_bin_num=10):
            self.name = 'Cart Pole'
            self.file_name = 'cart-pole-v0'
            self.env = gym.make('CartPole-v0')

            self.GAMMA = 1.0
            self.EPS_MIN = 0.0

            # State space analysis
            # observation = (cart x position, cart velocity, pole theta angle, pole velocity)
            # Num	Observation	            Min	        Max
            # 0	    Cart Position	        -2.4	    2.4
            # 1	    Cart Velocity	        -Inf	    Inf
            # 2	    Pole Angle	            ~ -41.8°	~ 41.8°
            # 3	    Pole Velocity At Tip	-Inf	    Inf

            # Discretize state space (10 bins each)                     # an example of bad modeling (won't converge):
            self.cartXSpace = np.linspace(-2.4, 2.4, 10)                                    # (-4.8, 4.8, 10)
            self.cartXVelSpace = np.linspace(-4, 4, 10)                                     # (-5, 5, 10)
            self.poleThetaSpace = np.linspace(-0.20943951, 0.20943951, pole_theta_bin_num)  # (-.418, .418, 10)
            self.poleThetaVelSpace = np.linspace(-4, 4, 10)                                 # (-5, 5, 10)

            self.single_state_space = single_state_space

            # Construct state space
            self.states = []
            if single_state_space == Envs_DSS.CartPole.CART_X:
                for i in range(len(self.cartXSpace) + 1):
                    self.states.append(i)
            elif single_state_space == Envs_DSS.CartPole.CART_X_VEL:
                for j in range(len(self.cartXVelSpace) + 1):
                    self.states.append(j)
            elif single_state_space == Envs_DSS.CartPole.POLE_THETA:
                for k in range(len(self.poleThetaSpace) + 1):
                    self.states.append(k)
            elif single_state_space == Envs_DSS.CartPole.POLE_THETA_VEL:
                for l in range(len(self.poleThetaVelSpace) + 1):
                    self.states.append(l)
            else:
                for i in range(len(self.cartXSpace) + 1):
                    for j in range(len(self.cartXVelSpace) + 1):
                        for k in range(len(self.poleThetaSpace) + 1):
                            for l in range(len(self.poleThetaVelSpace) + 1):
                                self.states.append((i, j, k, l))

        def get_state(self, observation):
            cart_x, cart_x_dot, pole_theta, pole_theta_dot = observation
            cart_x = int(np.digitize(cart_x, self.cartXSpace))
            cart_x_dot = int(np.digitize(cart_x_dot, self.cartXVelSpace))
            pole_theta = int(np.digitize(pole_theta, self.poleThetaSpace))
            pole_theta_dot = int(np.digitize(pole_theta_dot, self.poleThetaVelSpace))

            if self.single_state_space == Envs_DSS.CartPole.CART_X:
                return cart_x
            elif self.single_state_space == Envs_DSS.CartPole.CART_X_VEL:
                return cart_x_dot
            elif self.single_state_space == Envs_DSS.CartPole.POLE_THETA:
                return pole_theta
            elif self.single_state_space == Envs_DSS.CartPole.POLE_THETA_VEL:
                return pole_theta_dot
            else:
                return cart_x, cart_x_dot, pole_theta, pole_theta_dot

    class Acrobot:

        # The acrobot system includes two joints and two links, where the joint between the two links is actuated.
        # Initially, the links are hanging downwards.
        # the goal is to swing the end of the lower link up to a given height.

        # Acrobot-v1 is an unsolved environment, which means it does not have a specified reward threshold
        #   at which it's considered solved.
        # above -100 is prettty good. best score is: -42.37 ± 4.83

        # Actions (3):

        def __init__(self):
            self.name = 'Acrobot'
            self.file_name = 'acrobot-v1'
            self.env = gym.make('Acrobot-v1')

            self.GAMMA = 0.99
            self.EPS_MIN = 0.0

            # State space analysis
            # observation = (cos_theta1, sin_theta1, cos_theta2, sin_theta2, theta1_dot, theta2_dot)

            # Discretize state space (10 bins each)
            self.theta_space = np.linspace(-1, 1, 10)
            self.theta_dot_space = np.linspace(-10, 10, 10)

            # Construct state space
            self.states = []
            for c1 in range(len(self.theta_space) + 1):
                for s1 in range(len(self.theta_space) + 1):
                    for c2 in range(len(self.theta_space) + 1):
                        for s2 in range(len(self.theta_space) + 1):
                            for dot1 in range(len(self.theta_dot_space) + 1):
                                for dot2 in range(len(self.theta_dot_space) + 1):
                                    self.states.append((c1, s1, c2, s2, dot1, dot2))

        def get_state(self, observation):
            cos_theta1, sin_theta1, cos_theta2, sin_theta2, theta1_dot, theta2_dot = observation
            c_th1 = int(np.digitize(cos_theta1, self.theta_space))
            s_th1 = int(np.digitize(sin_theta1, self.theta_space))
            c_th2 = int(np.digitize(cos_theta2, self.theta_space))
            s_th2 = int(np.digitize(sin_theta2, self.theta_space))
            th1_dot = int(np.digitize(theta1_dot, self.theta_dot_space))
            th2_dot = int(np.digitize(theta2_dot, self.theta_dot_space))

            return c_th1, s_th1, c_th2, s_th2, th1_dot, th2_dot

    class MountainCar:

        # Starting State - Random position from -0.6 to -0.4 with no velocity.
        # Episode Termination - The episode ends when you reach 0.5 position, or if 200 iterations are reached.

        # Rewards:
        # -1 for each time step.
        # As with MountainCarContinuous v0, there is no penalty for climbing the left hill,
        #   which upon reached acts as a wall.

        # Actions (3): drive backward = left (0), do nothing (1), drive forward = right (2)

        CAR_POS = 0
        CAR_VEL = 1

        def __init__(self, single_state_space=-1, car_vel_bin_num=50):
            self.name = 'Mountain Car'
            self.file_name = 'mountain-car-v0'
            self.env = gym.make('MountainCar-v0')

            self.GAMMA = 1.0    # 0.99 (Q-learning) \ 1.0 (MC Policy Evaluation, TD-0)
            self.EPS_MIN = 0.0  # 0.01 (Q-learning) \ 0.0 (MC Policy Evaluation, TD-0)
                                # eps_max = 0.01 (Q-learning)

            # State space analysis
            # observation = (pos, vel)
            # Num	Observation	    Min	    Max
            # 0	    position	    -1.2	0.6
            # 1	    velocity	    -0.07	0.07

            # Discretize state space (into bins)
            self.carPosSpace = np.linspace(-1.2, 0.6, 9)  # (-1.2, 0.5, 8)
            self.carVelSpace = np.linspace(-0.07, 0.07, car_vel_bin_num)

            self.single_state_space = single_state_space

            # Construct state space
            self.states = []
            if single_state_space == Envs_DSS.MountainCar.CAR_POS:
                for pos in range(len(self.carPosSpace) + 1):
                    self.states.append(pos)
            elif single_state_space == Envs_DSS.MountainCar.CAR_VEL:
                for vel in range(len(self.carVelSpace) + 1):
                    self.states.append(vel)
            else:
                for pos in range(len(self.carPosSpace) + 1):
                    for vel in range(len(self.carVelSpace) + 1):
                        self.states.append((pos, vel))

        def get_state(self, observation):
            pos, vel = observation
            car_pos = int(np.digitize(pos, self.carPosSpace))  # pos_bin
            car_vel = int(np.digitize(vel, self.carVelSpace))  # vel_bin

            if self.single_state_space == Envs_DSS.MountainCar.CAR_POS:
                return car_pos
            elif self.single_state_space == Envs_DSS.MountainCar.CAR_VEL:
                return car_vel
            else:
                return car_pos, car_vel
