"""
https://github.com/openai/gym/wiki/Table-of-environments

https://github.com/openai/gym/tree/master/gym/envs -
    check gym/gym/envs/__init__.py for solved properties (max_episode_steps, reward_threshold, optimum).
    Solved: avg_score >= reward_threshold, over 100 consecutive trials.
    "Unsolved environment" - doesn't have a specified reward_threshold at which it's considered solved.

Environments with Discrete\Discretized State Space (DSS):
    Toy Text - FrozenLake, Taxi, Blackjack.  # TODO: add Grid World, Windy Gridworld, Custom Grid World.
    Classic Control - MountainCar, CartPole, Acrobot.
"""


import numpy as np
import gym


class BaseEnv:
    name: str
    file_name: str
    env: gym.wrappers.time_limit.TimeLimit
    states: list
    GAMMA: float
    EPS_MIN: float

    @staticmethod
    def get_evaluation_tuple():
        return None

    @staticmethod
    def update_evaluation_tuple(episode, reward, done, eval):
        return None

    @staticmethod
    def analyze_evaluation_tuple(eval, episodes):
        return None


########################################

# ToyText:

class FrozenLake(BaseEnv):
    """
    the agent walks on a 4x4 matrix.
    there's a single starting point (S), and a single goal (G) where the frisbee is located.
    the agent can walk on the frozen surface (F), but falls in the hole (H) to its doom.
    there's a chance for the agent to slip and not go in the chosen direction of the action.
    The episode ends when you reach the goal or fall in a hole.

    Reward: +1 when reaching the goal (0 otherwise)

    Solved:
    gym/gym/envs/__init__.py :
        FrozenLake-v0 (4x4 map): max_episode_steps = 100, reward_threshold = 0.78, optimum = .8196
        FrozenLake8x8-v0 (8x8 map): max_episode_steps = 200, reward_threshold = 0.99, optimum = 1

    Discrete observation space (1D).
        O = r * columns + c

    Discrete action space (1D).
    Actions (4): left (0), down (1), right (2), up (3)
    """

    def __init__(self):
        self.name = 'Frozen Lake'
        self.file_name = 'frozen-lake-v0'
        self.env = gym.make('FrozenLake-v0')

        # State space analysis:
        self.rows = 4
        self.columns = 4

        # Construct state space:
        self.states = []
        for r in range(self.rows):
            for c in range(self.columns):
                self.states.append((r, c))

        self.GAMMA = 0.9  # 0.99 ?
        self.EPS_MIN = 0.1

    def get_state(self, observation):
        """
        O = r * columns + c
        """
        c = observation % self.columns
        r = observation // self.columns
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
    """
    The taxi drives on a 5x5 matrix.

    Goal: pick up the passenger at one location and drop him off in another.
    Goal changes over time.

    There are:
        4 possible pick-up/drop-off destinations (R, G, B, Y).
        5 possible passenger locations (4 starting locations / destinations + Taxi).
    The pipe characters '|' indicate obstacles (the taxi cannot drive through them).

    Rewards:
        +20 for a successful drop-off.
        -1 for every time-step it takes.
        -10 for illegal pick-up and drop-off actions.
        -1 for driving against a wall.

    Solved:
        Taxi-v2 is an unsolved environment.
        100 Episodes Best Average Reward:
            leaders' board bottom (9.23) = reward_threshold
            leaders' board top (9.716) = optimum
    gym/gym/envs/__init__.py :
        Taxi-v3: max_episode_steps = 200, reward_threshold = 8, optimum = 8.46

    Discrete observation space (1D).
        O = ((tr * columns + tc) * passenger_locations + pl) * destinations + d.

    Discrete action space (1D).
    Actions (6): north (0), south (1), east (2), west (3), pick up (4), drop off (5)
    """

    def __init__(self):
        self.name = 'Taxi'
        self.file_name = 'taxi-v3'
        self.env = gym.make('Taxi-v3')

        # State space analysis:
        self.rows = 5
        self.columns = 5
        self.passenger_locations = 5
        self.destinations = 4

        # Construct state space:
        self.states = []
        for tr in range(self.rows):
            for tc in range(self.columns):
                for pl in range(self.passenger_locations):
                    for d in range(self.destinations):
                        self.states.append((tr, tc, pl, d))

        self.GAMMA = 0.999
        self.EPS_MIN = 0.0

    def get_state(self, o):
        """
        :param o: observation
        """
        d = o % self.destinations
        pl = (o // self.destinations) % self.passenger_locations
        tc = ((o // self.destinations) // self.passenger_locations) % self.columns
        tr = (((o // self.destinations) // self.passenger_locations) // self.columns)
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
    """
    At the start, the player receives two cards (so the total min is 2 + 2 = 4)
    Object: Have your card sum be greater than the dealers without exceeding 21.

    Reward: –1 for losing, 0 for a draw, and >=1 for winning (1.5 for natural = getting 21 on the first deal)

    Discrete observation space (3D).
        O = (sum, card, ace)
            sum - agent's cards sum (int) - the sum of the player's cards.
            card - dealer's showing card (int) - the card that the dealer has showing, 1 = Ace, 10 = face card.
            ace - agent's usable ace (bool) - if the player has usable ace, it can count as 1 / 11.

    Discrete action space (1D).
    Actions (2):
        stick = stop receiving cards (0)
        hit = receive another card (1)
    """

    def __init__(self):
        self.name = 'Blackjack'
        self.file_name = 'blackjack-v0'
        self.env = gym.make('Blackjack-v0')

        # State space analysis:
        self.agentCardsSumSpace = [i for i in range(4, 32)]  # was: range(4, 22), changed for e-SARSA
        self.dealerShowingCardSpace = [i + 1 for i in range(10)]
        self.agentUsableAceSpace = [False, True]

        # Construct state space:
        self.states = []
        for sum in self.agentCardsSumSpace:
            for card in self.dealerShowingCardSpace:
                for ace in self.agentUsableAceSpace:
                    self.states.append((sum, card, ace))

        self.GAMMA = 1.0
        self.EPS_MIN = 0.0

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


########################################

# ClassicControl:

class MountainCar(BaseEnv):
    """
    A car is on a one-dimensional track, positioned between two "mountains".

    Goal: to drive up the mountain on the right (reaching 0.5 position).
        however, the car's engine is not strong enough to scale the mountain in a single pass.
        Therefore, the only way to succeed is to drive back and forth to build up momentum.

    Starting State: Random position from -0.6 to -0.4 with no velocity.

    Episode Termination (besides reaching the goal): reaching 200 iterations.

    Rewards: -1 for each time-step.
        As with MountainCarContinuous v0, there is no penalty for climbing the left hill,
            which upon reached acts as a wall.
    No reward surrounding initial state.

    Solved:
    gym/gym/envs/__init__.py :
        MountainCar-v0: max_episode_steps = 200, reward_threshold = -110.0

    Continuous observation space (2D).
        O = ndarray[pos, vel]
            pos - position [-1.2, 0.6]
            vel - velocity [-0.07, 0.07]

    Discrete action space (1D).
    Actions (3): backward/left (0), none (1), forward/right (2)
    mountain_car_policy = lambda velocity_state: 0 if velocity_state < (car_vel_bin_num // 2) else 2
    """

    CAR_POS = 0
    CAR_VEL = 1

    def __init__(self, car_vel_bin_num=50, single_state_space=-1):
        self.name = 'Mountain Car'
        self.file_name = 'mountain-car-v0'
        self.env = gym.make('MountainCar-v0')

        # Discretize state space (into bins):
        self.carXSpace = np.linspace(-1.2, 0.6, 9)  # (-1.2, 0.5, 8)
        self.carVSpace = np.linspace(-0.07, 0.07, car_vel_bin_num)

        self.single_state_space = single_state_space

        # Construct state space (9*50=450):
        self.states = []
        if single_state_space == MountainCar.CAR_POS:
            for pos in range(len(self.carXSpace) + 1):
                self.states.append(pos)
        elif single_state_space == MountainCar.CAR_VEL:
            for vel in range(len(self.carVSpace) + 1):
                self.states.append(vel)
        else:
            for pos in range(len(self.carXSpace) + 1):
                for vel in range(len(self.carVSpace) + 1):
                    self.states.append((pos, vel))

        self.GAMMA = 1.0    # 0.99 (Q Learning) \ 1.0 (MC Policy Evaluation, TD-0)
        self.EPS_MIN = 0.0  # 0.01 (Q Learning) \ 0.0 (MC Policy Evaluation, TD-0)
                            # eps_max = 0.01 (Q Learning)

    def get_state(self, observation):
        x, x_dot = observation
        x_bin = int(np.digitize(x, self.carXSpace))
        x_dot_bin = int(np.digitize(x_dot, self.carVSpace))

        if self.single_state_space == MountainCar.CAR_POS:
            return x_bin
        elif self.single_state_space == MountainCar.CAR_VEL:
            return x_dot_bin
        else:
            return x_bin, x_dot_bin

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
    """
    AKA "Inverted Pendulum".
    A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track.
        The system is controlled by applying a force of +1 or -1 to the cart.

    Goal: to prevent the pendulum from falling over.

    Starting State: the pendulum starts upright.

    Episode Termination (besides reaching the goal):
        the pole is more than 15 degrees from vertical.
        the cart moves more than 2.4 units from the center.

    Rewards: +1 for every time-step that the pole remains upright.

    Solved:
    gym/gym/envs/__init__.py :
        CartPole-v0: max_episode_steps = 200, reward_threshold = 195.0
        CartPole-v1: max_episode_steps = 500, reward_threshold = 475.0

    Continuous observation space (4D).
        O = ndarray[x, x_dot, theta, theta_dot]
            x - Cart Position [-2.4, 2.4]
            x_dot - Cart Velocity [-Inf, Inf]
            theta - Pole Angle [~-41.8°, ~41.8°]
            theta_dot - Pole Velocity [-Inf, Inf]

    Discrete action space (1D).
    Actions (2): left (0), right (1)
    cart_pole_policy = lambda theta_state: 0 if theta_state < (pole_theta_bin_num // 2) else 1
    """

    CART_X = 0
    CART_V = 1
    POLE_THETA = 2
    POLE_V = 3

    def __init__(self, pole_theta_bin_num=10, single_state_space=-1):
        self.name = 'Cart Pole'
        self.file_name = 'cart-pole-v0'
        self.env = gym.make('CartPole-v0')

        # Discretize state space (10 bins each):                    # an example of bad modeling (won't converge):
        self.cartXSpace = np.linspace(-2.4, 2.4, 10)                                    # (-4.8, 4.8, 10)
        self.cartVSpace = np.linspace(-4, 4, 10)                                        # (-5, 5, 10)
        self.poleThetaSpace = np.linspace(-0.20943951, 0.20943951, pole_theta_bin_num)  # (-.418, .418, 10)
        self.poleVSpace = np.linspace(-4, 4, 10)                                        # (-5, 5, 10)

        self.single_state_space = single_state_space

        # Construct state space (10^4):
        self.states = []
        if single_state_space == CartPole.CART_X:
            for x in range(len(self.cartXSpace) + 1):
                self.states.append(x)
        elif single_state_space == CartPole.CART_V:
            for x_dot in range(len(self.cartVSpace) + 1):
                self.states.append(x_dot)
        elif single_state_space == CartPole.POLE_THETA:
            for theta in range(len(self.poleThetaSpace) + 1):
                self.states.append(theta)
        elif single_state_space == CartPole.POLE_V:
            for theta_dot in range(len(self.poleVSpace) + 1):
                self.states.append(theta_dot)
        else:
            for x in range(len(self.cartXSpace) + 1):
                for x_dot in range(len(self.cartVSpace) + 1):
                    for theta in range(len(self.poleThetaSpace) + 1):
                        for theta_dot in range(len(self.poleVSpace) + 1):
                            self.states.append((x, x_dot, theta, theta_dot))

        self.GAMMA = 1.0
        self.EPS_MIN = 0.0

    def get_state(self, observation):
        x, x_dot, theta, theta_dot = observation
        x_bin = int(np.digitize(x, self.cartXSpace))
        x_dot_bin = int(np.digitize(x_dot, self.cartVSpace))
        theta_bin = int(np.digitize(theta, self.poleThetaSpace))
        theta_dot_bin = int(np.digitize(theta_dot, self.poleVSpace))

        if self.single_state_space == CartPole.CART_X:
            return x_bin
        elif self.single_state_space == CartPole.CART_V:
            return x_dot_bin
        elif self.single_state_space == CartPole.POLE_THETA:
            return theta_bin
        elif self.single_state_space == CartPole.POLE_V:
            return theta_dot_bin
        else:
            return x_bin, x_dot_bin, theta_bin, theta_dot_bin


class Acrobot(BaseEnv):
    """
    The acrobot system includes two joints and two links, where the joint between the two links is actuated.

    Goal: to swing the end of the lower link up to a given height.

    Starting State: the links are hanging downwards.

    Episode Termination (besides reaching the goal): reaching 500 iterations.

    Rewards: -1 for each time-step the links are hanging downwards.
    No reward surrounding initial state.

    Solved:
    gym/gym/envs/__init__.py :
        Acrobot-v1: max_episode_steps = 500, reward_threshold = -100.0  # current best score: -42.37 ± 4.83

    Continuous observation space (6D).
        O = ndarray[cos_theta1, sin_theta1, cos_theta2, sin_theta2, theta1_dot, theta2_dot]

    Discrete action space (1D).
    Actions (3): applying torque on the joint between the two pendulum links:
        +1 torque (0)
        0 torque (1)
        -1 torque (2)
    """

    def __init__(self):
        self.name = 'Acrobot'
        self.file_name = 'acrobot-v1'
        self.env = gym.make('Acrobot-v1')

        # Discretize state space (10 bins each):
        self.theta_space = np.linspace(-1, 1, 10)
        self.theta_dot_space = np.linspace(-10, 10, 10)

        # Construct state space (10^6):
        self.states = []
        for c1 in range(len(self.theta_space) + 1):
            for s1 in range(len(self.theta_space) + 1):
                for c2 in range(len(self.theta_space) + 1):
                    for s2 in range(len(self.theta_space) + 1):
                        for dot1 in range(len(self.theta_dot_space) + 1):
                            for dot2 in range(len(self.theta_dot_space) + 1):
                                self.states.append((c1, s1, c2, s2, dot1, dot2))

        self.GAMMA = 0.99
        self.EPS_MIN = 0.0

    def get_state(self, observation):
        cos_theta1, sin_theta1, cos_theta2, sin_theta2, theta1_dot, theta2_dot = observation
        cos_theta1_bin = int(np.digitize(cos_theta1, self.theta_space))
        sin_theta1_bin = int(np.digitize(sin_theta1, self.theta_space))
        cos_theta2_bin = int(np.digitize(cos_theta2, self.theta_space))
        sin_theta2_bin = int(np.digitize(sin_theta2, self.theta_space))
        theta1_dot_bin = int(np.digitize(theta1_dot, self.theta_dot_space))
        theta2_dot_bin = int(np.digitize(theta2_dot, self.theta_dot_space))
        return cos_theta1_bin, sin_theta1_bin, cos_theta2_bin, sin_theta2_bin, theta1_dot_bin, theta2_dot_bin

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
