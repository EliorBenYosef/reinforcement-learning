import gym
import numpy as np
import matplotlib.pyplot as plt


# https://github.com/openai/gym/wiki/Table-of-environments

# Environments with Discretized State Space


class BaseEnv:

    @staticmethod
    def get_evaluation_tuple():
        return None

    @staticmethod
    def update_evaluation_tuple(episode, reward, done, eval):
        return None

    @staticmethod
    def analyze_evaluation_tuple(eval, episodes):
        return None


class Envs_DSS:

    # Toy Text - Grid World, Windy Gridworld

    class FrozenLake(BaseEnv):

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

        @staticmethod
        def get_evaluation_tuple():
            success = 0
            return success

        @staticmethod
        def update_evaluation_tuple(episode, reward, done, eval):
            success = eval
            if reward == 1:
                success += 1
            return success

        @staticmethod
        def analyze_evaluation_tuple(eval, episodes):
            success = eval
            print('success rate: %d%%' % (success * 100 / episodes))

    class Taxi(BaseEnv):

        # The taxi drives on a 5x5 matrix.

        # Goal: pick up the passenger at one location and drop him off in another.

        # There are:
        #   4 possible pick-up\drop-off destinations (R, G, B, Y).
        #   5 possible passenger locations (4 destinations +  Taxi).
        # The pipe characters '|' indicate obstacles (the taxi cannot drive through them).

        # Rewards:
        #   +20 for a successful drop-off.
        #   -1 for every time-step it takes.
        #   -10 for illegal pick-up and drop-off actions.
        #   -1 for driving against a wall.

        # Solved:
        #   This is an unsolved environment (doesn't have a specified reward threshold at which it's considered solved).
        #   100 Episodes Best Average Reward:
        #       top of the leader boars - 9.716
        #       bottom of the leader boars - 9.23

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

        @staticmethod
        def get_evaluation_tuple():
            successful_drop_offs = 0
            illegal_pick_up_or_drop_off = 0
            driving_against_a_wall = 0
            return successful_drop_offs, illegal_pick_up_or_drop_off, driving_against_a_wall

        @staticmethod
        def update_evaluation_tuple(episode, reward, done, eval):
            successful_drop_offs, illegal_pick_up_or_drop_off, driving_against_a_wall = eval
            if reward == 20:
                successful_drop_offs += 1
            elif reward == -2:
                illegal_pick_up_or_drop_off += 1
            elif reward == -10:
                driving_against_a_wall += 1
            return successful_drop_offs, illegal_pick_up_or_drop_off, driving_against_a_wall

        @staticmethod
        def analyze_evaluation_tuple(eval, episodes):
            successful_drop_offs, illegal_pick_up_or_drop_off, driving_against_a_wall = eval
            print('Rates - ',
                  'successful drop-offs: %d%% ;' % (successful_drop_offs * 100 / episodes),
                  'illegal pick-ups or drop-offs: %d%% ;' % (illegal_pick_up_or_drop_off * 100 / episodes),
                  'driving against a wall: %d%%' % (driving_against_a_wall * 100 / episodes))

    class Blackjack(BaseEnv):

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
            self.agentCardsSumSpace = [i for i in range(4, 32)]  # agent's cards sum - sum of the player's cards. was: range(4, 22), changed for e-SARSA
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

        @staticmethod
        def get_evaluation_tuple():
            wins = 0
            draws = 0
            losses = 0
            return wins, draws, losses

        @staticmethod
        def update_evaluation_tuple(episode, reward, done, eval):
            wins, draws, losses = eval
            if reward >= 1:
                wins += 1
            elif reward == 0:
                draws += 1
            elif reward == -1:
                losses += 1
            return wins, draws, losses

        @staticmethod
        def analyze_evaluation_tuple(eval, episodes):
            wins, draws, losses = eval
            print('Rates - ',
                  'win: %d%% ;' % (wins * 100 / episodes),
                  'draw: %d%% ;' % (losses * 100 / episodes),
                  'loss: %d%%' % (draws * 100 / episodes))

    # Custom Grid World

    # Classic Control

    class MountainCar(BaseEnv):

        # A car is on a one-dimensional track, positioned between two "mountains".

        # Goal: to drive up the mountain on the right (reaching 0.5 position).
        #   however, the car's engine is not strong enough to scale the mountain in a single pass.
        #   Therefore, the only way to succeed is to drive back and forth to build up momentum.

        # Starting State: Random position from -0.6 to -0.4 with no velocity.

        # Episode Termination (besides reaching the goal): reaching 200 iterations.

        # Rewards: -1 for each time-step.
        #   As with MountainCarContinuous v0, there is no penalty for climbing the left hill,
        #       which upon reached acts as a wall.

        # Solved: getting average reward of -110.0 over 100 consecutive trials.

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

            # Construct state space (450)
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

        @staticmethod
        def get_evaluation_tuple():
            success = 0
            return success

        @staticmethod
        def update_evaluation_tuple(episode, reward, done, eval):
            success = eval
            if done and episode < 200:
                success += 1
            return success

        @staticmethod
        def analyze_evaluation_tuple(eval, episodes):
            success = eval
            print('success rate: %d%%' % (success * 100 / episodes))

    class CartPole(BaseEnv):

        # AKA "Inverted Pendulum".
        # A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track.
        #   The system is controlled by applying a force of +1 or -1 to the cart.

        # Goal: to prevent the pendulum from falling over.

        # Starting State: the pendulum starts upright.

        # Episode Termination (besides reaching the goal):
        #   the pole is more than 15 degrees from vertical.
        #   the cart moves more than 2.4 units from the center.

        # Rewards: +1 for every time-step that the pole remains upright.

        # Solved: getting average reward of 195.0 over 100 consecutive trials.

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

            # Construct state space (10^4)
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

    class Acrobot(BaseEnv):

        # The acrobot system includes two joints and two links, where the joint between the two links is actuated.

        # Goal: to swing the end of the lower link up to a given height.

        # Starting State: the links are hanging downwards.

        # Episode Termination (besides reaching the goal): reaching 500 iterations.

        # Rewards: -1 for each time-step the links are hanging downwards.

        # Solved:
        #   This is an unsolved environment (doesn't have a specified reward threshold at which it's considered solved).
        #   above -100 is pretty good.
        #   best score is: -42.37 ± 4.83

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

            # Construct state space (10^6)
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

        @staticmethod
        def get_evaluation_tuple():
            success = 0
            return success

        @staticmethod
        def update_evaluation_tuple(episode, reward, done, eval):
            success = eval
            if done and reward == 0:
                success += 1
            return success

        @staticmethod
        def analyze_evaluation_tuple(eval, episodes):
            success = eval
            print('success rate: %d%%' % (success * 100 / episodes))
