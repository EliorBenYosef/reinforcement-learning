from numpy.random import seed
seed(28)

from reinforcement_learning.tabular_RL.envs_dss import FrozenLake, Taxi, Blackjack, MountainCar, CartPole, Acrobot
from reinforcement_learning.tabular_RL.algorithms.monte_carlo import MCPredictionModel, MCControlModel
from reinforcement_learning.tabular_RL.algorithms.td_zero import TD0PredictionModel, TD0ControlModel
from reinforcement_learning.tabular_RL.utils import test_q_table, test_policy_table
from reinforcement_learning.utils.plotter import plot_running_average, plot_accumulated_scores, \
    plot_running_average_comparison, plot_accumulated_scores_comparison


# Prediction (policy evaluation algorithms)

def test_mc_policy_evaluation(episodes=10, print_v_table=True):
    # Mountain Car:
    car_vel_bin_num = 34  # ~100% success rate (32 rarely loses)
    mountain_car_env = MountainCar(car_vel_bin_num, single_state_space=MountainCar.CAR_VEL)
    mountain_car_model = MCPredictionModel(mountain_car_env, episodes)
    mountain_car_model.perform_mc_policy_evaluation(
        policy=lambda velocity_state: 0 if velocity_state < (car_vel_bin_num // 2) else 2,  # mountain_car_policy
        print_info=print_v_table)

    # Cart Pole:
    pole_theta_bin_num = 10
    cart_pole_env = CartPole(pole_theta_bin_num, single_state_space=CartPole.POLE_THETA)
    cart_pole_model = MCPredictionModel(cart_pole_env, episodes=episodes)
    cart_pole_model.perform_mc_policy_evaluation(
        policy=lambda theta_state: 0 if theta_state < (pole_theta_bin_num // 2) else 1,  # cart_pole_policy
        print_info=print_v_table)

    # Frozen Lake:
    frozen_lake_env = FrozenLake()
    frozen_lake_model = MCPredictionModel(frozen_lake_env, episodes=episodes)
    frozen_lake_model.perform_mc_policy_evaluation(
        policy=lambda s: frozen_lake_env.env.action_space.sample(),  # frozen_lake_policy - random policy
        print_info=print_v_table)


def test_td0_policy_evaluation(episodes=10, print_v_table=True):
    # Mountain Car:
    car_vel_bin_num = 34  # ~100% success rate (32 rarely loses)
    mountain_car_env = MountainCar(car_vel_bin_num, single_state_space=MountainCar.CAR_VEL)
    mountain_car_model = TD0PredictionModel(mountain_car_env, episodes)
    mountain_car_model.perform_td0_policy_evaluation(
        policy=lambda velocity_state: 0 if velocity_state < (car_vel_bin_num // 2) else 2,  # mountain_car_policy
        print_info=print_v_table)

    # Cart Pole:
    pole_theta_bin_num = 10
    cart_pole_env = CartPole(pole_theta_bin_num, single_state_space=CartPole.POLE_THETA)
    cart_pole_model = TD0PredictionModel(cart_pole_env, episodes=episodes)
    cart_pole_model.perform_td0_policy_evaluation(
        policy=lambda theta_state: 0 if theta_state < (pole_theta_bin_num // 2) else 1,  # cart_pole_policy
        print_info=print_v_table)

    # Frozen Lake:
    frozen_lake_env = FrozenLake()
    frozen_lake_model = TD0PredictionModel(frozen_lake_env, episodes=episodes)
    frozen_lake_model.perform_td0_policy_evaluation(
        policy=lambda s: frozen_lake_env.env.action_space.sample(),  # frozen_lake_policy - random policy
        print_info=print_v_table)


# Control (value function estimation \ learning algorithms)

def test_mc_non_exploring_starts_control(episodes=100000, print_q_table_and_policy=True):
    method_name = 'MC non-exploring starts'

    # Frozen Lake:
    frozen_lake_env = FrozenLake()
    frozen_lake_model = MCControlModel(frozen_lake_env, episodes=episodes, eps_max=frozen_lake_env.EPS_MIN)
    frozen_lake_policy, frozen_lake_scores, frozen_lake_accumulated_scores = \
        frozen_lake_model.perform_mc_non_exploring_starts_control(print_info=print_q_table_and_policy)
    plot_running_average(frozen_lake_env.name, method_name, frozen_lake_scores, window=episodes//100, show=True)
    plot_accumulated_scores(frozen_lake_env.name, method_name, frozen_lake_accumulated_scores, show=True)
    frozen_lake_scores, frozen_lake_accumulated_scores = test_policy_table(frozen_lake_env, frozen_lake_policy)
    plot_running_average(frozen_lake_env.name, method_name, frozen_lake_scores, window=episodes//100, show=True)
    plot_accumulated_scores(frozen_lake_env.name, method_name, frozen_lake_accumulated_scores, show=True)

    # Blackjack:
    blackjack_env = Blackjack()
    blackjack_model = MCControlModel(blackjack_env, episodes=episodes, eps_max=0.05, eps_dec=1e-7)
    blackjack_policy, _, blackjack_accumulated_scores = \
        blackjack_model.perform_mc_non_exploring_starts_control(print_info=print_q_table_and_policy)
    plot_accumulated_scores(blackjack_env.name, method_name, blackjack_accumulated_scores, show=True)
    blackjack_accumulated_scores = test_policy_table(blackjack_env, blackjack_policy)
    plot_accumulated_scores(blackjack_env.name, method_name, blackjack_accumulated_scores, show=True)


def test_off_policy_mc_control(episodes=100000, print_q_table_and_policy=False):
    method_name = 'Off-policy MC Control'

    # Frozen Lake:
    frozen_lake_env = FrozenLake()
    frozen_lake_model = MCControlModel(frozen_lake_env, episodes=episodes, eps_max=frozen_lake_env.EPS_MIN)
    frozen_lake_policy, frozen_lake_scores, frozen_lake_accumulated_scores = \
        frozen_lake_model.perform_off_policy_mc_control(print_info=print_q_table_and_policy)
    plot_running_average(frozen_lake_env.name, method_name, frozen_lake_scores, window=episodes//100, show=True)
    plot_accumulated_scores(frozen_lake_env.name, method_name, frozen_lake_accumulated_scores, show=True)
    frozen_lake_scores, frozen_lake_accumulated_scores = test_policy_table(frozen_lake_env, frozen_lake_policy)
    plot_running_average(frozen_lake_env.name, method_name, frozen_lake_scores, window=episodes//100, show=True)
    plot_accumulated_scores(frozen_lake_env.name, method_name, frozen_lake_accumulated_scores, show=True)

    # Blackjack:
    blackjack_env = Blackjack()
    blackjack_model = MCControlModel(blackjack_env, episodes=episodes, eps_max=0.05, eps_dec=1e-7)
    blackjack_policy, _, blackjack_accumulated_scores = \
        blackjack_model.perform_off_policy_mc_control(print_info=print_q_table_and_policy)
    plot_accumulated_scores(blackjack_env.name, method_name, blackjack_accumulated_scores, show=True)
    blackjack_accumulated_scores = test_policy_table(blackjack_env, blackjack_policy)
    plot_accumulated_scores(blackjack_env.name, method_name, blackjack_accumulated_scores, show=True)


def test_sarsa():
    method_name = 'SARSA'

    # Taxi:
    taxi_env = Taxi()
    taxi_model = TD0ControlModel(taxi_env, episodes=10000, alpha=0.4)
    taxi_q_table, taxi_scores = taxi_model.perform_sarsa()
    plot_running_average(taxi_env.name, method_name, taxi_scores, show=True)
    taxi_scores = test_q_table(taxi_env, taxi_q_table)
    plot_running_average(taxi_env.name, method_name, taxi_scores, show=True)

    # Mountain Car:
    mountain_car_env = MountainCar()
    mountain_car_model = TD0ControlModel(mountain_car_env, episodes=50000)
    mountain_car_q_table, mountain_car_scores = mountain_car_model.perform_sarsa()
    plot_running_average(mountain_car_env.name, method_name, mountain_car_scores, show=True)
    mountain_car_scores = test_q_table(mountain_car_env, mountain_car_q_table)
    plot_running_average(mountain_car_env.name, method_name, mountain_car_scores, show=True)

    # Cart Pole (Solved):
    cart_pole_env = CartPole()
    cart_pole_model = TD0ControlModel(cart_pole_env, episodes=50000)
    cart_pole_q_table, cart_pole_scores = cart_pole_model.perform_sarsa()
    plot_running_average(cart_pole_env.name, method_name, cart_pole_scores, show=True)
    cart_pole_scores = test_q_table(cart_pole_env, cart_pole_q_table)
    plot_running_average(cart_pole_env.name, method_name, cart_pole_scores, show=True)

    # Acrobot:
    acrobot_env = Acrobot()
    acrobot_model = TD0ControlModel(acrobot_env, episodes=50000)
    acrobot_q_table, acrobot_scores = acrobot_model.perform_sarsa()
    plot_running_average(acrobot_env.name, method_name, acrobot_scores, show=True)
    acrobot_scores = test_q_table(acrobot_env, acrobot_q_table)
    plot_running_average(acrobot_env.name, method_name, acrobot_scores, show=True)


def test_expected_sarsa():
    method_name = 'Expected SARSA'

    # Taxi:
    taxi_env = Taxi()
    taxi_model = TD0ControlModel(taxi_env, episodes=10000, alpha=0.4)
    taxi_q_table, taxi_scores = taxi_model.perform_expected_sarsa()
    plot_running_average(taxi_env.name, method_name, taxi_scores, show=True)
    taxi_scores = test_q_table(taxi_env, taxi_q_table)
    plot_running_average(taxi_env.name, method_name, taxi_scores, show=True)

    # Mountain Car:
    mountain_car_env = MountainCar()
    mountain_car_model = TD0ControlModel(mountain_car_env, episodes=50000)
    mountain_car_q_table, mountain_car_scores = mountain_car_model.perform_expected_sarsa()
    plot_running_average(mountain_car_env.name, method_name, mountain_car_scores, show=True)
    mountain_car_scores = test_q_table(mountain_car_env, mountain_car_q_table)
    plot_running_average(mountain_car_env.name, method_name, mountain_car_scores, show=True)

    # Cart Pole (Solved):
    cart_pole_env = CartPole()
    cart_pole_model = TD0ControlModel(cart_pole_env, episodes=50000)
    cart_pole_q_table, cart_pole_scores = cart_pole_model.perform_expected_sarsa()
    plot_running_average(cart_pole_env.name, method_name, cart_pole_scores, show=True)
    cart_pole_scores = test_q_table(cart_pole_env, cart_pole_q_table)
    plot_running_average(cart_pole_env.name, method_name, cart_pole_scores, show=True)

    # Acrobot:
    acrobot_env = Acrobot()
    acrobot_model = TD0ControlModel(acrobot_env, episodes=50000)
    acrobot_q_table, acrobot_scores = acrobot_model.perform_expected_sarsa()
    plot_running_average(acrobot_env.name, method_name, acrobot_scores, show=True)
    acrobot_scores = test_q_table(acrobot_env, acrobot_q_table)
    plot_running_average(acrobot_env.name, method_name, acrobot_scores, show=True)


def test_q_learning():
    method_name = 'Q-learning'

    # Taxi:
    taxi_env = Taxi()
    taxi_model = TD0ControlModel(taxi_env, episodes=10000, alpha=0.4)
    taxi_q_table, taxi_scores = taxi_model.perform_q_learning()
    plot_running_average(taxi_env.name, method_name, taxi_scores, show=True)
    taxi_scores = test_q_table(taxi_env, taxi_q_table)
    plot_running_average(taxi_env.name, method_name, taxi_scores, show=True)

    # Mountain Car:
    mountain_car_env = MountainCar()
    mountain_car_model = TD0ControlModel(mountain_car_env, episodes=50000)
    mountain_car_q_table, mountain_car_scores = mountain_car_model.perform_q_learning()
    plot_running_average(mountain_car_env.name, method_name, mountain_car_scores, show=True)
    mountain_car_scores = test_q_table(mountain_car_env, mountain_car_q_table)
    plot_running_average(mountain_car_env.name, method_name, mountain_car_scores, show=True)

    # Cart Pole:
    cart_pole_env = CartPole()
    cart_pole_model = TD0ControlModel(cart_pole_env, episodes=50000)
    cart_pole_q_table, cart_pole_scores = cart_pole_model.perform_q_learning()
    plot_running_average(cart_pole_env.name, method_name, cart_pole_scores, show=True)
    cart_pole_scores = test_q_table(cart_pole_env, cart_pole_q_table)
    plot_running_average(cart_pole_env.name, method_name, cart_pole_scores, show=True)


def test_double_q_learning():
    method_name = 'Double Q-learning'

    # Taxi:
    taxi_env = Taxi()
    taxi_model = TD0ControlModel(taxi_env, episodes=10000, alpha=0.4)
    taxi_q1_table, taxi_q2_table, taxi_scores = taxi_model.perform_double_q_learning()
    plot_running_average(taxi_env.name, method_name, taxi_scores, show=True)
    taxi_q1_scores = test_q_table(taxi_env, taxi_q1_table)
    taxi_q2_scores = test_q_table(taxi_env, taxi_q2_table)
    scores_list = [taxi_q1_scores, taxi_q2_scores]
    labels = ['Q1', 'Q2']
    plot_running_average_comparison(taxi_env.name + ' - ' + method_name, scores_list, labels, show=True)

    # Mountain Car:
    mountain_car_env = MountainCar()
    mountain_car_model = TD0ControlModel(mountain_car_env, episodes=50000)
    mountain_car_q1_table, mountain_car_q2_table, mountain_car_scores = \
        mountain_car_model.perform_double_q_learning()
    plot_running_average(mountain_car_env.name, method_name, mountain_car_scores, show=True)
    mountain_car_q1_scores = test_q_table(mountain_car_env, mountain_car_q1_table)
    mountain_car_q2_scores = test_q_table(mountain_car_env, mountain_car_q2_table)
    scores_list = [mountain_car_q1_scores, mountain_car_q2_scores]
    labels = ['Q1', 'Q2']
    plot_running_average_comparison(mountain_car_env.name + ' - ' + method_name, scores_list, labels, show=True)

    # Cart Pole:
    cart_pole_env = CartPole()
    cart_pole_model = TD0ControlModel(cart_pole_env, episodes=50000)
    cart_pole_q1_table, cart_pole_q2_table, cart_pole_scores = \
        cart_pole_model.perform_double_q_learning()
    plot_running_average(cart_pole_env.name, method_name, cart_pole_scores, show=True)
    cart_pole_q1_scores = test_q_table(cart_pole_env, cart_pole_q1_table)
    cart_pole_q2_scores = test_q_table(cart_pole_env, cart_pole_q2_table)
    scores_list = [cart_pole_q1_scores, cart_pole_q2_scores]
    labels = ['Q1', 'Q2']
    plot_running_average_comparison(cart_pole_env.name + ' - ' + method_name, scores_list, labels, show=True)


# Environments

def environment_test(env, episodes, eps_max=1.0, eps_dec=None, alpha=0.1,
                     q_table_test_method=test_q_table,
                     policy_test_method=test_policy_table,
                     show_scores=True, show_accumulated_scores=True):

    labels = [
        'MC non-exploring starts',
        'off-policy MC',
        'SARSA',
        'Expected SARSA',
        'Q-learning',
        'Double Q-learning'
    ]

    mc_model_01 = MCControlModel(env, episodes=episodes, alpha=alpha, eps_max=eps_max, eps_dec=eps_dec)
    policy_mc_01, scores_mc_01, accumulated_scores_mc_01 = mc_model_01.perform_mc_non_exploring_starts_control()

    mc_model_02 = MCControlModel(env, episodes=episodes, alpha=alpha, eps_max=eps_max, eps_dec=eps_dec)
    policy_mc_02, scores_mc_02, accumulated_scores_mc_02 = mc_model_02.perform_off_policy_mc_control()

    sarsa_model = TD0ControlModel(env, episodes=episodes, alpha=alpha, eps_max=eps_max, eps_dec=eps_dec)
    q_table_sarsa, scores_sarsa, accumulated_scores_sarsa = sarsa_model.perform_sarsa()

    e_sarsa_model = TD0ControlModel(env, episodes=episodes, alpha=alpha, eps_max=eps_max, eps_dec=eps_dec)
    q_table_e_sarsa, scores_e_sarsa, accumulated_scores_e_sarsa = e_sarsa_model.perform_expected_sarsa()

    q_l_model = TD0ControlModel(env, episodes=episodes, alpha=alpha, eps_max=eps_max, eps_dec=eps_dec)
    q_table_q_l, scores_q_l, accumulated_scores_q_l = q_l_model.perform_q_learning()

    d_q_l_model = TD0ControlModel(env, episodes=episodes, alpha=alpha, eps_max=eps_max, eps_dec=eps_dec)
    q1_table_d_q_l, q2_table_d_q_l, scores_d_q_l, accumulated_scores_d_q_l = d_q_l_model.perform_double_q_learning()
    q_table_d_q_l = {}
    for s in q1_table_d_q_l:
        q_table_d_q_l[s] = (q1_table_d_q_l[s] + q2_table_d_q_l[s]) / 2

    if show_scores:
        scores_list = [scores_mc_01, scores_mc_02, scores_sarsa, scores_e_sarsa, scores_q_l, scores_d_q_l]
        plot_running_average_comparison(env.name + ' - Training', scores_list, labels,  # window=episodes//100,
                                        file_name=env.file_name + '-score-training')

    if show_accumulated_scores:
        accumulated_scores_list = [accumulated_scores_mc_01, accumulated_scores_mc_02,
                                   accumulated_scores_sarsa, accumulated_scores_e_sarsa,
                                   accumulated_scores_q_l, accumulated_scores_d_q_l]
        plot_accumulated_scores_comparison(env.name + ' - Training', accumulated_scores_list, labels,
                                           file_name=env.file_name + '-accumulated-score-training')

    scores_mc_01, accumulated_scores_mc_01 = policy_test_method(env, policy_mc_01)
    scores_mc_02, accumulated_scores_mc_02 = policy_test_method(env, policy_mc_02)
    scores_sarsa, accumulated_scores_sarsa = q_table_test_method(env, q_table_sarsa)
    scores_e_sarsa, accumulated_scores_e_sarsa = q_table_test_method(env, q_table_e_sarsa)
    scores_q_l, accumulated_scores_q_l = q_table_test_method(env, q_table_q_l)
    scores_d_q_l, accumulated_scores_d_q_l = q_table_test_method(env, q_table_d_q_l)

    if show_scores:
        scores_list = [scores_mc_01, scores_mc_02, scores_sarsa, scores_e_sarsa, scores_q_l, scores_d_q_l]
        plot_running_average_comparison(env.name + ' - Test', scores_list, labels,  # window=episodes//100,
                                        file_name=env.file_name + '-scores-test')

    if show_accumulated_scores:
        accumulated_scores_list = [accumulated_scores_mc_01, accumulated_scores_mc_02,
                                   accumulated_scores_sarsa, accumulated_scores_e_sarsa,
                                   accumulated_scores_q_l, accumulated_scores_d_q_l]
        plot_accumulated_scores_comparison(env.name + ' - Test', accumulated_scores_list, labels,
                                           file_name=env.file_name + '-accumulated-scores-test')


def test_environments():
    environment_test(FrozenLake(), episodes=100000, eps_max=0.1, eps_dec=None)
    environment_test(Taxi(), episodes=10000, alpha=0.4)
    environment_test(Blackjack(), episodes=100000, eps_max=0.05, eps_dec=1e-7)
    environment_test(MountainCar(), episodes=50000)
    environment_test(CartPole(), episodes=50000)
    environment_test(Acrobot(), episodes=50000)
