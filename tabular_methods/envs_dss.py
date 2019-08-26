import numpy as np
import gym


# https://github.com/openai/gym/wiki/Table-of-environments

# Environments with Discretized State Space

class Envs_DSS:

    # Toy Text - Grid World, Windy Gridworld

    class FrozenLake:

        # the agent walks on a 4x4 matrix.
        # there's a single starting point (S), and a single goal (G) where the frisbee is located.
        # the agent can walk on the frozen surface (F), but falls in the hole (H) to its doom.
        # there's a chance for the agent to slip and not go in the chosen direction of the action.

        # track accumulated reward (better)

        def __init__(self):
            self.env = gym.make('FrozenLake-v0')  # env.action_space.n = 4
            self.file_name = 'frozen-lake-v0'

            # 0 = left, 1 = down, 2 = right, 3 = up

            # Set discount rate (gamma)
            self.GAMMA = 0.9  # 0.99 ?

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

        @staticmethod
        def get_action(s, s_):
            a = -1
            rs, cs = s
            rs_, cs_ = s_

            if cs != cs_:
                if cs_ < cs:
                    a = 0  # left
                elif cs_ > cs:
                    a = 2  # right
            elif rs != rs_:
                if rs_ < rs:
                    a = 3  # up
                elif rs_ > rs:
                    a = 1  # down

            return a

    class Taxi:

        # The taxi drives on a 5x5 matrix.
        # There are 5 possible passenger locations (R, G, Y, B, and Taxi), and 4 possible destinations.
        # The pipe characters '|' indicate obstacles: the taxi cannot drive through them.

        def __init__(self):
            self.env = gym.make('Taxi-v2')  # env.action_space.n = 4
            self.file_name = 'taxi-v2'

            # 0 = left, 1 = down, 2 = right, 3 = up

            # Set discount rate (gamma)
            self.GAMMA = 0.999  # 0.999 ?

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

        # track accumulated reward (better)

        def __init__(self):
            self.env = gym.make('Blackjack-v0')  # env.action_space.n = 2
            self.file_name = 'blackjack-v0'

            # 0 = stick (stop receiving cards), 1 = hit (receive another card)

            # Set discount rate (gamma)
            #   in blackjack the rewards are certain
            self.GAMMA = 1.0

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

    # Custom Grid World

    # Classic Control

    class CartPole:

        def __init__(self, single_state_space=-1):
            self.env = gym.make('CartPole-v0')  # env.action_space.n = 2
            self.file_name = 'cart-pole-v0'

            # Set discount rate (gamma)
            #   Shouldn't be discounted, because in CartPole future rewards are certain:
            #   the state transition functions are deterministic, the state transition probabilities are 1.
            #   you know where the pole and cart are gonna move.
            self.GAMMA = 1.0

            # State space analysis
            # observation = (cart x position, cart velocity, pole theta angle, pole velocity)
            # Num	Observation	            Min	        Max
            # 0	    Cart Position	        -2.4	    2.4
            # 1	    Cart Velocity	        -Inf	    Inf
            # 2	    Pole Angle	            ~ -41.8°	~ 41.8°
            # 3	    Pole Velocity At Tip	-Inf	    Inf

            # Discretize state space (10 bins each)                         # an example of bad modeling (won't converge):
            self.cartXSpace = np.linspace(-2.4, 2.4, 10)                    # (-4.8, 4.8, 10)
            self.cartXVelSpace = np.linspace(-4, 4, 10)                     # (-5, 5, 10)
            self.poleThetaSpace = np.linspace(-0.20943951, 0.20943951, 10)  # (-.418, .418, 10)
            self.poleThetaVelSpace = np.linspace(-4, 4, 10)                 # (-5, 5, 10)

            self.single_state_space = single_state_space

            # Construct state space
            self.states = []
            if single_state_space == 0:
                for i in range(len(self.cartXSpace) + 1):
                    self.states.append(i)
            elif single_state_space == 1:
                for j in range(len(self.cartXVelSpace) + 1):
                    self.states.append(j)
            elif single_state_space == 2:
                for k in range(len(self.poleThetaSpace) + 1):
                    self.states.append(k)
            elif single_state_space == 3:
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

            if self.single_state_space == 0:
                return cart_x
            elif self.single_state_space == 1:
                return cart_x_dot
            elif self.single_state_space == 2:
                return pole_theta
            elif self.single_state_space == 3:
                return pole_theta_dot
            else:
                return cart_x, cart_x_dot, pole_theta, pole_theta_dot

    class Acrobot:

        def __init__(self):
            self.env = gym.make('Acrobot-v1')  # env.action_space.n = 3
            self.file_name = 'acrobot-v1'

            # Set discount rate (gamma)
            self.GAMMA = 0.99

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

        def __init__(self, single_state_space=-1):
            self.env = gym.make('MountainCar-v0')  # env.action_space.n = 3
            self.file_name = 'mountain-car-v0'

            # Set discount rate (gamma)
            self.GAMMA = 0.99  # 1.0 ?

            # State space analysis
            # observation = (pos, vel)
            # Num	Observation	    Min	    Max
            # 0	    position	    -1.2	0.6
            # 1	    velocity	    -0.07	0.07

            # Discretize state space
            self.carPosSpace = np.linspace(-1.2, 0.6, 9)  # posBins, (-1.2, 0.5, 8)
            self.carVelSpace = np.linspace(-0.07, 0.07, 50)  # velBins, (-0.07, 0.07, 8)

            self.single_state_space = single_state_space

            # Construct state space
            self.states = []
            if single_state_space == 0:
                for pos in range(len(self.carPosSpace) + 1):
                    self.states.append(pos)
            elif single_state_space == 1:
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

            if self.single_state_space == 0:
                return car_pos
            elif self.single_state_space == 1:
                return car_vel
            else:
                return car_pos, car_vel

